// Copyright 2026 Google
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::mem::size_of;

use crate::util::AsBorrowedDescriptor;
use crate::util::Error;
use crate::util::Handle;
use crate::util::OwnedDescriptor;
use crate::util::Reader;
use crate::util::Result;
use crate::util::Tube;
use crate::util::Writer;
use crate::util::MAGMA_GPU_HANDLE_TYPE_SIGNAL_EVENT_FD;
use zerocopy::FromBytes;
use zerocopy::Immutable;
use zerocopy::IntoBytes;

use crate::protocols::kumquat_gpu_protocol::*;

const MAX_COMMAND_SIZE: usize = 65536;

fn expected_message_size(buf: &[u8]) -> Option<usize> {
    if buf.len() < size_of::<kumquat_gpu_protocol_ctrl_hdr>() {
        return None;
    }
    let hdr = kumquat_gpu_protocol_ctrl_hdr::read_from_prefix(buf).ok()?.0;
    match hdr.type_ {
        KUMQUAT_GPU_PROTOCOL_GET_NUM_CAPSETS
        | KUMQUAT_GPU_PROTOCOL_GET_CAPSET_INFO
        | KUMQUAT_GPU_PROTOCOL_CTX_DESTROY
        | KUMQUAT_GPU_PROTOCOL_SNAPSHOT_SAVE
        | KUMQUAT_GPU_PROTOCOL_SNAPSHOT_RESTORE
        | KUMQUAT_GPU_PROTOCOL_RESP_NUM_CAPSETS
        | KUMQUAT_GPU_PROTOCOL_RESP_CONTEXT_CREATE
        | KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT => Some(size_of::<kumquat_gpu_protocol_ctrl_hdr>()),
        KUMQUAT_GPU_PROTOCOL_GET_CAPSET => Some(size_of::<kumquat_gpu_protocol_get_capset>()),
        KUMQUAT_GPU_PROTOCOL_CTX_CREATE => Some(size_of::<kumquat_gpu_protocol_ctx_create>()),
        KUMQUAT_GPU_PROTOCOL_CTX_ATTACH_RESOURCE
        | KUMQUAT_GPU_PROTOCOL_CTX_DETACH_RESOURCE => {
            Some(size_of::<kumquat_gpu_protocol_ctx_resource>())
        }
        KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_3D => {
            Some(size_of::<kumquat_gpu_protocol_resource_create_3d>())
        }
        KUMQUAT_GPU_PROTOCOL_TRANSFER_TO_HOST_3D
        | KUMQUAT_GPU_PROTOCOL_TRANSFER_FROM_HOST_3D => {
            Some(size_of::<kumquat_gpu_protocol_transfer_host_3d>())
        }
        KUMQUAT_GPU_PROTOCOL_SUBMIT_3D => {
            if buf.len() < size_of::<kumquat_gpu_protocol_cmd_submit>() {
                return None;
            }
            let cmd = kumquat_gpu_protocol_cmd_submit::read_from_prefix(buf).ok()?.0;
            let fence_bytes = (cmd.num_in_fences as usize) * size_of::<u64>();
            let cmd_bytes = cmd.size as usize;
            Some(size_of::<kumquat_gpu_protocol_cmd_submit>() + fence_bytes + cmd_bytes)
        }
        KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_BLOB => {
            Some(size_of::<kumquat_gpu_protocol_resource_create_blob>())
        }
        KUMQUAT_GPU_PROTOCOL_RESP_CAPSET_INFO => {
            Some(size_of::<kumquat_gpu_protocol_resp_capset_info>())
        }
        KUMQUAT_GPU_PROTOCOL_RESP_CAPSET => {
            Some(size_of::<kumquat_gpu_protocol_ctrl_hdr>() + (hdr.payload as usize))
        }
        KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE => {
            Some(size_of::<kumquat_gpu_protocol_resp_resource_create>())
        }
        KUMQUAT_GPU_PROTOCOL_RESP_CMD_SUBMIT_3D => {
            Some(size_of::<kumquat_gpu_protocol_resp_cmd_submit_3d>())
        }
        _ => Some(size_of::<kumquat_gpu_protocol_ctrl_hdr>()),
    }
}

pub struct KumquatStream {
    stream: Tube,
    write_buffer: Vec<u8>,
    read_buffer: Vec<u8>,
    pending_data: Vec<u8>,
    pending_descriptors: VecDeque<OwnedDescriptor>,
}

impl KumquatStream {
    pub fn new(stream: Tube) -> KumquatStream {
        KumquatStream {
            stream,
            write_buffer: vec![0; MAX_COMMAND_SIZE],
            read_buffer: vec![0; MAX_COMMAND_SIZE],
            pending_data: Vec::new(),
            pending_descriptors: VecDeque::new(),
        }
    }

    pub fn write<T: FromBytes + IntoBytes + Immutable>(
        &mut self,
        encode: KumquatGpuProtocolWrite<T>,
    ) -> Result<()> {
        let needed_len = match &encode {
            KumquatGpuProtocolWrite::Cmd(_) | KumquatGpuProtocolWrite::CmdWithHandle(_, _) => {
                size_of::<T>()
            }
            KumquatGpuProtocolWrite::CmdWithData(_, data) => size_of::<T>() + data.len(),
        };
        if needed_len > self.write_buffer.len() {
            self.write_buffer.resize(needed_len, 0);
        }

        let mut writer = Writer::new(&mut self.write_buffer);

        let array: Vec<OwnedDescriptor> = match encode {
            KumquatGpuProtocolWrite::Cmd(cmd) => {
                writer.write_obj(cmd)?;
                Vec::new()
            }
            KumquatGpuProtocolWrite::CmdWithHandle(cmd, handle) => {
                writer.write_obj(cmd)?;
                vec![handle.os_handle]
            }
            KumquatGpuProtocolWrite::CmdWithData(cmd, data) => {
                writer.write_obj(cmd)?;
                writer.write_all(&data)?;
                Vec::new()
            }
        };

        let bytes_written = writer.bytes_written();
        let res = self
            .stream
            .send(&self.write_buffer[0..bytes_written], array);
        if self.write_buffer.len() > MAX_COMMAND_SIZE {
            self.write_buffer.resize(MAX_COMMAND_SIZE, 0);
            self.write_buffer.shrink_to_fit();
        }
        res?;
        Ok(())
    }

    pub fn read(&mut self) -> Result<Vec<KumquatGpuProtocol>> {
        let mut vec: Vec<KumquatGpuProtocol> = Vec::new();

        let complete_bytes = loop {
            let mut complete = 0;
            while complete < self.pending_data.len() {
                match expected_message_size(&self.pending_data[complete..]) {
                    Some(msg_size) if complete + msg_size <= self.pending_data.len() => {
                        complete += msg_size;
                    }
                    _ => break,
                }
            }
            if complete > 0 {
                break complete;
            }

            let (bytes_read, descriptor_vec) = self.stream.receive(&mut self.read_buffer)?;
            if bytes_read == 0 {
                if self.pending_data.is_empty() {
                    vec.push(KumquatGpuProtocol::OkNoData);
                    return Ok(vec);
                } else {
                    return Err(Error::Unsupported);
                }
            }
            self.pending_data
                .extend_from_slice(&self.read_buffer[0..bytes_read]);
            self.pending_descriptors.extend(descriptor_vec);
        };

        let mut reader = Reader::new(&self.pending_data[0..complete_bytes]);
        while reader.available_bytes() != 0 {
            let hdr = reader.peek_obj::<kumquat_gpu_protocol_ctrl_hdr>()?;
            let protocol = match hdr.type_ {
                KUMQUAT_GPU_PROTOCOL_GET_NUM_CAPSETS => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::GetNumCapsets
                }
                KUMQUAT_GPU_PROTOCOL_GET_CAPSET_INFO => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::GetCapsetInfo(hdr.payload)
                }
                KUMQUAT_GPU_PROTOCOL_GET_CAPSET => {
                    KumquatGpuProtocol::GetCapset(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_CTX_CREATE => {
                    KumquatGpuProtocol::CtxCreate(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_CTX_DESTROY => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::CtxDestroy(hdr.payload)
                }
                KUMQUAT_GPU_PROTOCOL_CTX_ATTACH_RESOURCE => {
                    KumquatGpuProtocol::CtxAttachResource(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_CTX_DETACH_RESOURCE => {
                    KumquatGpuProtocol::CtxDetachResource(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_3D => {
                    KumquatGpuProtocol::ResourceCreate3d(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_TRANSFER_TO_HOST_3D => {
                    let os_handle = self.pending_descriptors.pop_front().ok_or(Error::Unsupported)?;
                    let resp: kumquat_gpu_protocol_transfer_host_3d = reader.read_obj()?;

                    let handle = Handle {
                        os_handle,
                        handle_type: MAGMA_GPU_HANDLE_TYPE_SIGNAL_EVENT_FD,
                    };

                    KumquatGpuProtocol::TransferToHost3d(resp, handle)
                }
                KUMQUAT_GPU_PROTOCOL_TRANSFER_FROM_HOST_3D => {
                    let os_handle = self.pending_descriptors.pop_front().ok_or(Error::Unsupported)?;
                    let resp: kumquat_gpu_protocol_transfer_host_3d = reader.read_obj()?;

                    let handle = Handle {
                        os_handle,
                        handle_type: MAGMA_GPU_HANDLE_TYPE_SIGNAL_EVENT_FD,
                    };

                    KumquatGpuProtocol::TransferFromHost3d(resp, handle)
                }
                KUMQUAT_GPU_PROTOCOL_SUBMIT_3D => {
                    let cmd: kumquat_gpu_protocol_cmd_submit = reader.read_obj()?;
                    let num_in_fences = cmd.num_in_fences as usize;
                    let cmd_size = cmd.size as usize;
                    let mut cmd_buf = vec![0; cmd_size];
                    let mut fence_ids: Vec<u64> = Vec::with_capacity(num_in_fences);
                    for _ in 0..num_in_fences {
                        match reader.read_obj::<u64>() {
                            Ok(fence_id) => {
                                fence_ids.push(fence_id);
                            }
                            Err(_) => return Err(Error::Unsupported),
                        }
                    }
                    if cmd_size > 0 {
                        reader.read_exact(&mut cmd_buf[..])?;
                    }
                    KumquatGpuProtocol::CmdSubmit3d(cmd, cmd_buf, fence_ids)
                }
                KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_BLOB => {
                    KumquatGpuProtocol::ResourceCreateBlob(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_SNAPSHOT_SAVE => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::SnapshotSave
                }
                KUMQUAT_GPU_PROTOCOL_SNAPSHOT_RESTORE => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::SnapshotRestore
                }
                KUMQUAT_GPU_PROTOCOL_RESP_NUM_CAPSETS => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::RespNumCapsets(hdr.payload)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_CAPSET_INFO => {
                    KumquatGpuProtocol::RespCapsetInfo(reader.read_obj()?)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_CAPSET => {
                    let len: usize = hdr.payload.try_into()?;
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    let mut capset: Vec<u8> = vec![0; len];
                    reader.read_exact(&mut capset)?;
                    KumquatGpuProtocol::RespCapset(capset)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_CONTEXT_CREATE => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::RespContextCreate(hdr.payload)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_RESOURCE_CREATE => {
                    let os_handle = self.pending_descriptors.pop_front().ok_or(Error::Unsupported)?;
                    let resp: kumquat_gpu_protocol_resp_resource_create = reader.read_obj()?;

                    let handle = Handle {
                        os_handle,
                        handle_type: resp.handle_type,
                    };

                    KumquatGpuProtocol::RespResourceCreate(resp, handle)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_CMD_SUBMIT_3D => {
                    let os_handle = self.pending_descriptors.pop_front().ok_or(Error::Unsupported)?;
                    let resp: kumquat_gpu_protocol_resp_cmd_submit_3d = reader.read_obj()?;

                    let handle = Handle {
                        os_handle,
                        handle_type: resp.handle_type,
                    };

                    KumquatGpuProtocol::RespCmdSubmit3d(resp.fence_id, handle)
                }
                KUMQUAT_GPU_PROTOCOL_RESP_OK_SNAPSHOT => {
                    reader.consume(size_of::<kumquat_gpu_protocol_ctrl_hdr>());
                    KumquatGpuProtocol::RespOkSnapshot
                }
                _ => {
                    return Err(Error::Unsupported);
                }
            };

            vec.push(protocol);
        }

        self.pending_data.drain(0..complete_bytes);
        Ok(vec)
    }

    pub fn as_borrowed_descriptor(&self) -> &OwnedDescriptor {
        self.stream.as_borrowed_descriptor()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::thread;

    use rustix::fs::fcntl_setfl;
    use rustix::fs::OFlags;
    use rustix::net::socketpair;
    use rustix::net::AddressFamily;
    use rustix::net::SocketFlags;
    use rustix::net::SocketType;
    use zerocopy::IntoBytes;

    use super::*;

    fn create_stream_pair(nonblocking: bool) -> (Tube, Tube) {
        let (fd_a, fd_b) = socketpair(
            AddressFamily::UNIX,
            SocketType::STREAM,
            SocketFlags::CLOEXEC,
            None,
        )
        .unwrap();
        if nonblocking {
            fcntl_setfl(&fd_a, OFlags::NONBLOCK).unwrap();
            fcntl_setfl(&fd_b, OFlags::NONBLOCK).unwrap();
        }
        (
            Tube::try_from(OwnedDescriptor::from(fd_a)).unwrap(),
            Tube::try_from(OwnedDescriptor::from(fd_b)).unwrap(),
        )
    }

    #[test]
    fn stream_reassembles_fragmented_and_coalesced_messages() {
        let (sender_tube, receiver_tube) = create_stream_pair(false);
        let mut reader = KumquatStream::new(receiver_tube);

        let submit_hdr = kumquat_gpu_protocol_cmd_submit {
            hdr: kumquat_gpu_protocol_ctrl_hdr {
                type_: KUMQUAT_GPU_PROTOCOL_SUBMIT_3D,
                payload: 0,
            },
            ctx_id: 7,
            size: 8,
            num_in_fences: 1,
            ..Default::default()
        };
        let fence_id: u64 = 42;
        let payload: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        let mut raw_msg = Vec::new();
        raw_msg.extend_from_slice(submit_hdr.as_bytes());
        raw_msg.extend_from_slice(fence_id.as_bytes());
        raw_msg.extend_from_slice(&payload);

        let destroy_hdr = kumquat_gpu_protocol_ctrl_hdr {
            type_: KUMQUAT_GPU_PROTOCOL_CTX_DESTROY,
            payload: 7,
        };

        let handle = thread::spawn(move || {
            // Send first 5 bytes (partial header), then remainder of header + payload + coalesced CtxDestroy
            sender_tube.send(&raw_msg[..5], Vec::new()).unwrap();
            let mut rest = raw_msg[5..].to_vec();
            rest.extend_from_slice(destroy_hdr.as_bytes());
            sender_tube.send(&rest, Vec::new()).unwrap();
        });

        let mut received = reader.read().unwrap();
        if received.len() == 1 {
            received.extend(reader.read().unwrap());
        }
        handle.join().unwrap();

        assert_eq!(received.len(), 2);
        match &received[0] {
            KumquatGpuProtocol::CmdSubmit3d(cmd, data, fences) => {
                assert_eq!(cmd.ctx_id, 7);
                assert_eq!(fences, &vec![42u64]);
                assert_eq!(data, &payload);
            }
            other => panic!("unexpected first protocol message: {other:?}"),
        }
        match &received[1] {
            KumquatGpuProtocol::CtxDestroy(ctx_id) => assert_eq!(*ctx_id, 7),
            other => panic!("unexpected second protocol message: {other:?}"),
        }
    }

    #[test]
    fn large_payload_grows_write_buffer_and_polls_on_nonblocking_stream() {
        let (sender_tube, receiver_tube) = create_stream_pair(true);
        let mut writer = KumquatStream::new(sender_tube);
        let mut reader = KumquatStream::new(receiver_tube);

        // 256 KiB exceeds MAX_COMMAND_SIZE (64 KiB) and standard AF_UNIX socket buffer sizes,
        // exercising both dynamic write_buffer growth/shrink and EAGAIN poll() loops.
        let large_payload: Vec<u8> = (0..(256 * 1024)).map(|i| (i % 251) as u8).collect();
        let expected_payload = large_payload.clone();

        let reader_thread = thread::spawn(move || {
            let protocols = reader.read().unwrap();
            assert_eq!(protocols.len(), 1);
            match &protocols[0] {
                KumquatGpuProtocol::RespCapset(data) => assert_eq!(data, &expected_payload),
                other => panic!("unexpected protocol message: {other:?}"),
            }
        });

        let hdr = kumquat_gpu_protocol_ctrl_hdr {
            type_: KUMQUAT_GPU_PROTOCOL_RESP_CAPSET,
            payload: large_payload.len() as u32,
        };
        writer
            .write(KumquatGpuProtocolWrite::CmdWithData(hdr, large_payload))
            .unwrap();
        assert_eq!(writer.write_buffer.len(), MAX_COMMAND_SIZE);

        reader_thread.join().unwrap();
    }
}

