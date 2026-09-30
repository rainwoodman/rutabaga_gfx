// Copyright 2024 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap as Map;
use std::path::PathBuf;

use magma_gpu::util::AsBorrowedDescriptor;
use magma_gpu::util::Error as MagmaGpuError;
use magma_gpu::util::Listener;
use magma_gpu::util::WaitContext;
use magma_gpu::util::WaitTimeout;

use crate::kumquat_gpu::KumquatGpu;
use crate::kumquat_gpu::KumquatGpuConnection;
use crate::kumquat_gpu::KumquatGpuResult;

enum KumquatConnection {
    GpuListener,
    GpuConnection(Box<KumquatGpuConnection>),
}

pub struct Kumquat {
    connection_id: u64,
    wait_ctx: WaitContext,
    kumquat_gpu_opt: Option<KumquatGpu>,
    gpu_listener_opt: Option<Listener>,
    connections: Map<u64, KumquatConnection>,
}

impl Kumquat {
    pub fn run(&mut self) -> KumquatGpuResult<()> {
        let events = self.wait_ctx.wait(WaitTimeout::NoTimeout)?;
        for event in events {
            let mut hung_up = false;
            match self.connections.entry(event.connection_id) {
                Entry::Occupied(mut o) => {
                    let connection = o.get_mut();
                    match connection {
                        KumquatConnection::GpuListener => {
                            if let Some(ref listener) = self.gpu_listener_opt {
                                let stream = listener.accept()?;
                                self.connection_id += 1;
                                let new_gpu_conn = KumquatGpuConnection::new(stream);
                                self.wait_ctx.add(
                                    self.connection_id,
                                    new_gpu_conn.as_borrowed_descriptor(),
                                )?;
                                self.connections.insert(
                                    self.connection_id,
                                    KumquatConnection::GpuConnection(Box::new(new_gpu_conn)),
                                );
                            }
                        }
                        KumquatConnection::GpuConnection(ref mut gpu_conn) => {
                            if event.readable {
                                if let Some(ref mut kumquat_gpu) = self.kumquat_gpu_opt {
                                    match gpu_conn.process_command(kumquat_gpu) {
                                        Ok(cmd_hung_up) => {
                                            hung_up = cmd_hung_up;
                                        }
                                        Err(e) => {
                                            log::warn!("kumquat gpu connection closed with error: {:?}", e);
                                            hung_up = true;
                                        }
                                    }
                                }
                            } else if event.hung_up {
                                hung_up = true;
                            }

                            if hung_up {
                                if let Some(ref mut kumquat_gpu) = self.kumquat_gpu_opt {
                                    gpu_conn.cleanup(kumquat_gpu);
                                }
                                self.wait_ctx.delete(gpu_conn.as_borrowed_descriptor())?;
                                o.remove_entry();
                            }
                        }
                    }
                }
                Entry::Vacant(_) => {
                    return Err(MagmaGpuError::WithContext("no connection found").into())
                }
            }
        }

        Ok(())
    }
}

pub struct KumquatBuilder {
    capset_names_opt: Option<String>,
    gpu_socket_opt: Option<String>,
    renderer_features_opt: Option<String>,
}

impl KumquatBuilder {
    pub fn new() -> KumquatBuilder {
        KumquatBuilder {
            capset_names_opt: None,
            gpu_socket_opt: None,
            renderer_features_opt: None,
        }
    }

    pub fn set_capset_names(mut self, capset_names: String) -> KumquatBuilder {
        self.capset_names_opt = Some(capset_names);
        self
    }

    pub fn set_gpu_socket(mut self, gpu_socket_opt: Option<String>) -> KumquatBuilder {
        self.gpu_socket_opt = gpu_socket_opt;
        self
    }

    pub fn set_renderer_features(mut self, renderer_features: String) -> KumquatBuilder {
        self.renderer_features_opt = Some(renderer_features);
        self
    }

    pub fn build(self) -> KumquatGpuResult<Kumquat> {
        let connection_id: u64 = 0;
        let mut wait_ctx = WaitContext::new()?;
        let mut kumquat_gpu_opt: Option<KumquatGpu> = None;
        let mut gpu_listener_opt: Option<Listener> = None;
        let mut connections: Map<u64, KumquatConnection> = Default::default();

        if let Some(gpu_socket) = self.gpu_socket_opt {
            // Remove path if it exists
            let path = PathBuf::from(&gpu_socket);
            let _ = std::fs::remove_file(&path);

            // Should not panic, since main.rs always calls set_capset_names and
            // set_renderer_features, even with the empty string.
            kumquat_gpu_opt = Some(KumquatGpu::new(
                self.capset_names_opt.unwrap(),
                self.renderer_features_opt.unwrap(),
            )?);

            let gpu_listener = Listener::bind(path)?;
            wait_ctx.add(connection_id, gpu_listener.as_borrowed_descriptor())?;
            connections.insert(connection_id, KumquatConnection::GpuListener);
            gpu_listener_opt = Some(gpu_listener);
        }

        Ok(Kumquat {
            connection_id,
            wait_ctx,
            kumquat_gpu_opt,
            gpu_listener_opt,
            connections,
        })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use magma_gpu::protocols::ipc::KumquatStream;
    use magma_gpu::protocols::kumquat_gpu_protocol::*;
    use magma_gpu::util::Tube;
    use magma_gpu::util::TubeType;
    use rutabaga_gfx::RUTABAGA_CAPSET_CROSS_DOMAIN;

    use super::*;

    #[test]
    fn client_error_and_disconnect_isolated_and_cleans_up_resources() {
        let sock_path = std::env::temp_dir()
            .join(format!("kumquat-test-{}.sock", std::process::id()))
            .to_string_lossy()
            .into_owned();

        let mut server = KumquatBuilder::new()
            .set_capset_names("cross-domain".to_string())
            .set_gpu_socket(Some(sock_path.clone()))
            .set_renderer_features("SystemBlob:enabled".to_string())
            .build()
            .unwrap();

        let mut client1 = KumquatStream::new(Tube::new(&sock_path, TubeType::Packet).unwrap());
        server.run().unwrap();
        assert_eq!(server.connections.len(), 2);

        // Create a cross-domain context
        client1
            .write(KumquatGpuProtocolWrite::Cmd(
                kumquat_gpu_protocol_ctx_create {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_CTX_CREATE,
                        payload: 0,
                    },
                    nlen: 0,
                    context_init: RUTABAGA_CAPSET_CROSS_DOMAIN,
                    debug_name: [0; 64],
                },
            ))
            .unwrap();
        server.run().unwrap();
        let resp = client1.read().unwrap();
        let ctx_id = match resp.as_slice() {
            [KumquatGpuProtocol::RespContextCreate(id)] => *id,
            other => panic!("expected RespContextCreate, got {other:?}"),
        };

        // Create a 3D resource attached to ctx_id
        client1
            .write(KumquatGpuProtocolWrite::Cmd(
                kumquat_gpu_protocol_resource_create_3d {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_RESOURCE_CREATE_3D,
                        payload: 0,
                    },
                    target: 2,
                    format: 1,
                    bind: 2,
                    width: 64,
                    height: 4,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: 0,
                    size: 4096,
                    stride: 256,
                    ctx_id,
                },
            ))
            .unwrap();
        server.run().unwrap();
        let _ = client1.read().unwrap();
        assert_eq!(
            server.kumquat_gpu_opt.as_ref().unwrap().resources.len(),
            1
        );

        // Drop client1 abruptly without detaching the resource or destroying the context
        drop(client1);
        server.run().unwrap();

        // Connection is removed and its attached resource/context are cleaned up
        assert_eq!(server.connections.len(), 1);
        assert!(server.kumquat_gpu_opt.as_ref().unwrap().resources.is_empty());

        // Server remains healthy and accepts a new client connection
        let mut client2 = KumquatStream::new(Tube::new(&sock_path, TubeType::Packet).unwrap());
        server.run().unwrap();
        client2
            .write(KumquatGpuProtocolWrite::Cmd(
                kumquat_gpu_protocol_ctrl_hdr {
                    type_: KUMQUAT_GPU_PROTOCOL_GET_NUM_CAPSETS,
                    payload: 0,
                },
            ))
            .unwrap();
        server.run().unwrap();
        let resp2 = client2.read().unwrap();
        assert!(matches!(
            resp2.as_slice(),
            [KumquatGpuProtocol::RespNumCapsets(1)]
        ));

        // Sending a command that errors in process_command() (detaching a non-existent resource)
        // also isolates the error to client2 without failing server.run().
        client2
            .write(KumquatGpuProtocolWrite::Cmd(
                kumquat_gpu_protocol_ctx_resource {
                    hdr: kumquat_gpu_protocol_ctrl_hdr {
                        type_: KUMQUAT_GPU_PROTOCOL_CTX_DETACH_RESOURCE,
                        payload: 0,
                    },
                    ctx_id: 999,
                    resource_id: 999,
                },
            ))
            .unwrap();
        server.run().unwrap();
        assert_eq!(server.connections.len(), 1);

        let _ = std::fs::remove_file(&sock_path);
    }
}

