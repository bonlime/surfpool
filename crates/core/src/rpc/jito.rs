use std::str::FromStr;

use jsonrpc_core::{BoxFuture, Error, Result};
use jsonrpc_derive::rpc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use solana_client::{
    rpc_config::RpcSendTransactionConfig, rpc_custom_error::RpcCustomError,
    rpc_response::RpcResponseContext,
};
use solana_commitment_config::CommitmentConfig;
use solana_rpc_client_api::response::Response as RpcResponse;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;
use solana_transaction_error::TransactionError;
use solana_transaction_status::UiTransactionEncoding;
use surfpool_types::TransactionStatusEvent;

use super::{
    RunloopContext, State, SurfnetRpcContext,
    full::{Full, SurfpoolFullRpc, SurfpoolRpcSendTransactionConfig},
    utils::decode_and_deserialize,
};
use crate::surfnet::locker::SurfnetSvmLocker;

/// Jito-specific RPC methods for bundle submission and simulation.
#[rpc]
pub trait Jito {
    type Metadata;

    /// Sends a bundle of transactions to be processed sequentially.
    ///
    /// This RPC method accepts a bundle of transactions (Jito-compatible format) and processes
    /// them one by one in order. All transactions in the bundle must succeed for the bundle to
    /// be accepted.
    ///
    /// ## Parameters
    /// - `transactions`: An array of serialized transaction data (base64 or base58 encoded).
    /// - `config`: Optional configuration for encoding format.
    ///
    /// ## Returns
    /// - `Result<String>`: A bundle ID (SHA-256 hash of comma-separated signatures).
    #[rpc(meta, name = "sendBundle")]
    fn send_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        config: Option<RpcSendTransactionConfig>,
    ) -> Result<String>;

    /// Simulates a bundle of transactions sequentially without mutating the live Surfpool state.
    ///
    /// ## Parameters
    /// - `params[0].encodedTransactions`: Serialized transactions.
    /// - `params[1].transactionEncoding` (optional): `base64` or `base58` (default: `base64`).
    ///
    /// ## Returns
    /// A standard Solana `RpcResponse` with a `summary` field:
    /// - `"succeeded"` when all transactions pass
    /// - `{ "failed": { "error": ..., "tx_signature": ... } }` on first failure
    #[rpc(meta, name = "simulateBundle")]
    fn simulate_bundle(
        &self,
        meta: Self::Metadata,
        params: RpcSimulateBundleParams,
        config: Option<RpcSimulateBundleConfig>,
    ) -> BoxFuture<Result<RpcResponse<RpcSimulateBundleValue>>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSimulateBundleParams {
    pub encoded_transactions: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSimulateBundleConfig {
    #[serde(default)]
    pub transaction_encoding: Option<UiTransactionEncoding>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RpcSimulateBundleValue {
    pub summary: RpcSimulateBundleSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum RpcSimulateBundleSummary {
    Succeeded(String),
    Failed(RpcSimulateBundleFailedSummary),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RpcSimulateBundleFailedSummary {
    pub failed: RpcSimulateBundleFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RpcSimulateBundleFailed {
    pub error: serde_json::Value,
    pub tx_signature: Option<String>,
}

#[derive(Clone)]
pub struct SurfpoolJitoRpc;

impl Jito for SurfpoolJitoRpc {
    type Metadata = Option<RunloopContext>;

    fn send_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        config: Option<RpcSendTransactionConfig>,
    ) -> Result<String> {
        if transactions.is_empty() {
            return Err(Error::invalid_params("Bundle cannot be empty"));
        }

        let Some(_ctx) = &meta else {
            return Err(RpcCustomError::NodeUnhealthy {
                num_slots_behind: None,
            }
            .into());
        };

        let full_rpc = SurfpoolFullRpc;
        let mut bundle_signatures = Vec::new();

        // Process each transaction in the bundle sequentially using Full RPC.
        // Force skip_preflight to match Jito Block Engine behavior (no simulation on sendBundle).
        for (idx, tx_data) in transactions.iter().enumerate() {
            let base_config = config.clone().unwrap_or_default();
            let bundle_config = Some(SurfpoolRpcSendTransactionConfig {
                base: RpcSendTransactionConfig {
                    skip_preflight: true,
                    ..base_config
                },
                skip_sig_verify: None,
            });

            match full_rpc.send_transaction(meta.clone(), tx_data.clone(), bundle_config) {
                Ok(signature_str) => {
                    let signature = Signature::from_str(&signature_str).map_err(|e| {
                        Error::invalid_params(format!("Failed to parse signature: {e}"))
                    })?;
                    bundle_signatures.push(signature);
                }
                Err(e) => {
                    return Err(Error {
                        code: e.code,
                        message: format!("Bundle transaction {} failed: {}", idx, e.message),
                        data: e.data,
                    });
                }
            }
        }

        // Calculate bundle ID by hashing comma-separated signatures (Jito-compatible).
        // https://github.com/jito-foundation/jito-solana/blob/master/sdk/src/bundle/mod.rs#L21
        use sha2::{Digest, Sha256};
        let concatenated_signatures = bundle_signatures
            .iter()
            .map(|sig| sig.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut hasher = Sha256::new();
        hasher.update(concatenated_signatures.as_bytes());
        let bundle_id = hasher.finalize();
        Ok(hex::encode(bundle_id))
    }

    fn simulate_bundle(
        &self,
        meta: Self::Metadata,
        params: RpcSimulateBundleParams,
        config: Option<RpcSimulateBundleConfig>,
    ) -> BoxFuture<Result<RpcResponse<RpcSimulateBundleValue>>> {
        let Some(_ctx) = &meta else {
            return Box::pin(async {
                Err(RpcCustomError::NodeUnhealthy {
                    num_slots_behind: None,
                }
                .into())
            });
        };

        if params.encoded_transactions.is_empty() {
            return Box::pin(async { Err(Error::invalid_params("Bundle cannot be empty")) });
        }

        let tx_encoding = config
            .as_ref()
            .and_then(|cfg| cfg.transaction_encoding)
            .unwrap_or(UiTransactionEncoding::Base64);
        let binary_encoding = match tx_encoding.into_binary_encoding() {
            Some(binary_encoding) => binary_encoding,
            None => {
                return Box::pin(async move {
                    Err(Error::invalid_params(format!(
                        "unsupported transactionEncoding: {tx_encoding}. Supported encodings: base58, base64"
                    )))
                });
            }
        };

        let SurfnetRpcContext {
            svm_locker,
            remote_ctx,
        } = match meta.get_rpc_context(CommitmentConfig::confirmed()) {
            Ok(res) => res,
            Err(e) => return e.into(),
        };

        Box::pin(async move {
            let context_slot = svm_locker.get_latest_absolute_slot();
            let simulate_locker = clone_locker_for_simulation(&svm_locker);

            let mut transactions = Vec::with_capacity(params.encoded_transactions.len());
            for encoded_transaction in params.encoded_transactions {
                let (_, tx) = decode_and_deserialize::<VersionedTransaction>(
                    encoded_transaction,
                    binary_encoding,
                )?;
                transactions.push(tx);
            }

            for tx in transactions {
                let tx_signature = tx.signatures.first().map(|sig| sig.to_string());
                let (status_update_tx, status_update_rx) = crossbeam_channel::bounded(1);

                if let Err(e) = simulate_locker
                    .process_transaction(
                        &remote_ctx,
                        tx,
                        status_update_tx,
                        false, // keep preflight enabled for simulation semantics
                        true,  // do signature verification by default
                    )
                    .await
                {
                    return Ok(RpcResponse {
                        context: RpcResponseContext::new(context_slot),
                        value: RpcSimulateBundleValue {
                            summary: RpcSimulateBundleSummary::Failed(
                                RpcSimulateBundleFailedSummary {
                                    failed: RpcSimulateBundleFailed {
                                        error: json!({
                                            "message": format!("Failed to process transaction: {e}")
                                        }),
                                        tx_signature,
                                    },
                                },
                            ),
                        },
                    });
                }

                match status_update_rx.recv() {
                    Ok(TransactionStatusEvent::Success(_)) => {}
                    Ok(TransactionStatusEvent::SimulationFailure((error, metadata))) => {
                        return Ok(RpcResponse {
                            context: RpcResponseContext::new(context_slot),
                            value: RpcSimulateBundleValue {
                                summary: failure_summary_from_tx_error(
                                    &error,
                                    &metadata.logs,
                                    tx_signature,
                                ),
                            },
                        });
                    }
                    Ok(TransactionStatusEvent::ExecutionFailure((error, metadata))) => {
                        return Ok(RpcResponse {
                            context: RpcResponseContext::new(context_slot),
                            value: RpcSimulateBundleValue {
                                summary: failure_summary_from_tx_error(
                                    &error,
                                    &metadata.logs,
                                    tx_signature,
                                ),
                            },
                        });
                    }
                    Ok(TransactionStatusEvent::VerificationFailure(signature)) => {
                        return Ok(RpcResponse {
                            context: RpcResponseContext::new(context_slot),
                            value: RpcSimulateBundleValue {
                                summary: RpcSimulateBundleSummary::Failed(
                                    RpcSimulateBundleFailedSummary {
                                        failed: RpcSimulateBundleFailed {
                                            error: json!({ "verificationError": signature }),
                                            tx_signature,
                                        },
                                    },
                                ),
                            },
                        });
                    }
                    Err(recv_error) => {
                        return Ok(RpcResponse {
                            context: RpcResponseContext::new(context_slot),
                            value: RpcSimulateBundleValue {
                                summary: RpcSimulateBundleSummary::Failed(
                                    RpcSimulateBundleFailedSummary {
                                        failed: RpcSimulateBundleFailed {
                                            error: json!({
                                                "message": format!(
                                                    "Failed to receive transaction status: {recv_error}"
                                                )
                                            }),
                                            tx_signature,
                                        },
                                    },
                                ),
                            },
                        });
                    }
                }
            }

            Ok(RpcResponse {
                context: RpcResponseContext::new(context_slot),
                value: RpcSimulateBundleValue {
                    summary: RpcSimulateBundleSummary::Succeeded("succeeded".to_string()),
                },
            })
        })
    }
}

fn clone_locker_for_simulation(svm_locker: &SurfnetSvmLocker) -> SurfnetSvmLocker {
    let simulation_svm = svm_locker.with_svm_reader(|svm_reader| svm_reader.clone_for_profiling());
    SurfnetSvmLocker::new(simulation_svm)
}

fn failure_summary_from_tx_error(
    error: &TransactionError,
    logs: &[String],
    tx_signature: Option<String>,
) -> RpcSimulateBundleSummary {
    RpcSimulateBundleSummary::Failed(RpcSimulateBundleFailedSummary {
        failed: RpcSimulateBundleFailed {
            error: json!({
                "transactionError": error.to_string(),
                "logs": logs,
            }),
            tx_signature,
        },
    })
}

#[cfg(test)]
mod tests {
    use base64::{Engine as _, prelude::BASE64_STANDARD};
    use jsonrpc_core::ErrorCode;
    use sha2::{Digest, Sha256};
    use solana_keypair::Keypair;
    use solana_message::{VersionedMessage, v0::Message as V0Message};
    use solana_pubkey::Pubkey;
    use solana_signer::Signer;
    use solana_system_interface::{instruction as system_instruction, program as system_program};
    use solana_transaction::versioned::VersionedTransaction;
    use solana_transaction_status::TransactionBinaryEncoding;
    use surfpool_types::{SimnetCommand, TransactionConfirmationStatus, TransactionStatusEvent};

    use super::*;
    use crate::{
        tests::helpers::TestSetup,
        types::{SurfnetTransactionStatus, TransactionWithStatusMeta},
    };

    const LAMPORTS_PER_SOL: u64 = 1_000_000_000;

    fn build_v0_transaction(
        payer: &Pubkey,
        signers: &[&Keypair],
        instructions: &[solana_instruction::Instruction],
        recent_blockhash: &solana_hash::Hash,
    ) -> VersionedTransaction {
        let msg = VersionedMessage::V0(
            V0Message::try_compile(payer, instructions, &[], *recent_blockhash).unwrap(),
        );
        VersionedTransaction::try_new(msg, signers).unwrap()
    }

    fn encode_transaction(
        tx: &VersionedTransaction,
        encoding: TransactionBinaryEncoding,
    ) -> String {
        let bytes = bincode::serialize(tx).unwrap();
        match encoding {
            TransactionBinaryEncoding::Base58 => bs58::encode(bytes).into_string(),
            TransactionBinaryEncoding::Base64 => BASE64_STANDARD.encode(bytes),
        }
    }

    #[test]
    fn test_send_bundle_empty_bundle_rejected() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let result = setup.rpc.send_bundle(Some(setup.context), vec![], None);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Bundle cannot be empty"),
            "Expected 'Bundle cannot be empty' error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_send_bundle_no_context_returns_unhealthy() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let result = setup
            .rpc
            .send_bundle(None, vec!["some_tx".to_string()], None);

        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_send_bundle_single_transaction() {
        let payer = Keypair::new();
        let recipient = Pubkey::new_unique();
        let (mempool_tx, mempool_rx) = crossbeam_channel::unbounded();
        let setup = TestSetup::new_with_mempool(SurfpoolJitoRpc, mempool_tx);
        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        // Airdrop to payer
        let _ = setup
            .context
            .svm_locker
            .0
            .write()
            .await
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL);

        let tx = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient,
                LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );
        let tx_encoded = bs58::encode(bincode::serialize(&tx).unwrap()).into_string();
        let expected_sig = tx.signatures[0];

        let setup_clone = setup.clone();
        let handle = hiro_system_kit::thread_named("send_bundle")
            .spawn(move || {
                setup_clone
                    .rpc
                    .send_bundle(Some(setup_clone.context), vec![tx_encoded], None)
            })
            .unwrap();

        // Process the transaction from mempool
        loop {
            match mempool_rx.recv() {
                Ok(SimnetCommand::ProcessTransaction(_, tx, status_tx, _, _)) => {
                    let mut writer = setup.context.svm_locker.0.write().await;
                    let slot = writer.get_latest_absolute_slot();
                    writer.transactions_queued_for_confirmation.push_back((
                        tx.clone(),
                        status_tx.clone(),
                        None,
                    ));
                    let sig = tx.signatures[0];
                    let tx_with_status_meta = TransactionWithStatusMeta {
                        slot,
                        transaction: tx,
                        ..Default::default()
                    };
                    let mutated_accounts = std::collections::HashSet::new();
                    writer
                        .transactions
                        .store(
                            sig.to_string(),
                            SurfnetTransactionStatus::processed(
                                tx_with_status_meta,
                                mutated_accounts,
                            ),
                        )
                        .unwrap();
                    status_tx
                        .send(TransactionStatusEvent::Success(
                            TransactionConfirmationStatus::Confirmed,
                        ))
                        .unwrap();
                    break;
                }
                Ok(SimnetCommand::AirdropProcessed) => continue,
                _ => panic!("failed to receive transaction from mempool"),
            }
        }

        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Bundle should succeed: {:?}", result);

        // Verify bundle ID is SHA-256 of the signature
        let bundle_id = result.unwrap();
        let mut hasher = Sha256::new();
        hasher.update(expected_sig.to_string().as_bytes());
        let expected_bundle_id = hex::encode(hasher.finalize());
        assert_eq!(
            bundle_id, expected_bundle_id,
            "Bundle ID should match SHA-256 of signature"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_send_bundle_multiple_transactions() {
        let payer = Keypair::new();
        let recipient1 = Pubkey::new_unique();
        let recipient2 = Pubkey::new_unique();
        let (mempool_tx, mempool_rx) = crossbeam_channel::unbounded();
        let setup = TestSetup::new_with_mempool(SurfpoolJitoRpc, mempool_tx);
        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        // Airdrop to payer
        let _ = setup
            .context
            .svm_locker
            .0
            .write()
            .await
            .airdrop(&payer.pubkey(), 5 * LAMPORTS_PER_SOL);

        let tx1 = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient1,
                LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );
        let tx2 = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient2,
                LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );

        let tx1_encoded = bs58::encode(bincode::serialize(&tx1).unwrap()).into_string();
        let tx2_encoded = bs58::encode(bincode::serialize(&tx2).unwrap()).into_string();
        let expected_sig1 = tx1.signatures[0];
        let expected_sig2 = tx2.signatures[0];

        let setup_clone = setup.clone();
        let handle = hiro_system_kit::thread_named("send_bundle")
            .spawn(move || {
                setup_clone.rpc.send_bundle(
                    Some(setup_clone.context),
                    vec![tx1_encoded, tx2_encoded],
                    None,
                )
            })
            .unwrap();

        // Process both transactions from mempool
        let mut processed_count = 0;
        while processed_count < 2 {
            match mempool_rx.recv() {
                Ok(SimnetCommand::ProcessTransaction(_, tx, status_tx, _, _)) => {
                    let mut writer = setup.context.svm_locker.0.write().await;
                    let slot = writer.get_latest_absolute_slot();
                    writer.transactions_queued_for_confirmation.push_back((
                        tx.clone(),
                        status_tx.clone(),
                        None,
                    ));
                    let sig = tx.signatures[0];
                    let tx_with_status_meta = TransactionWithStatusMeta {
                        slot,
                        transaction: tx,
                        ..Default::default()
                    };
                    let mutated_accounts = std::collections::HashSet::new();
                    writer
                        .transactions
                        .store(
                            sig.to_string(),
                            SurfnetTransactionStatus::processed(
                                tx_with_status_meta,
                                mutated_accounts,
                            ),
                        )
                        .unwrap();
                    status_tx
                        .send(TransactionStatusEvent::Success(
                            TransactionConfirmationStatus::Confirmed,
                        ))
                        .unwrap();
                    processed_count += 1;
                }
                Ok(SimnetCommand::AirdropProcessed) => continue,
                _ => panic!("failed to receive transaction from mempool"),
            }
        }

        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Bundle should succeed: {:?}", result);

        // Verify bundle ID is SHA-256 of comma-separated signatures
        let bundle_id = result.unwrap();
        let concatenated = format!("{},{}", expected_sig1, expected_sig2);
        let mut hasher = Sha256::new();
        hasher.update(concatenated.as_bytes());
        let expected_bundle_id = hex::encode(hasher.finalize());
        assert_eq!(
            bundle_id, expected_bundle_id,
            "Bundle ID should match SHA-256 of comma-separated signatures"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_empty_bundle_rejected() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let err = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![],
                },
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(err.message.contains("Bundle cannot be empty"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_success_base58() {
        let payer = Keypair::new();
        let recipient = Pubkey::new_unique();
        let setup = TestSetup::new(SurfpoolJitoRpc);

        let _ = setup
            .context
            .svm_locker
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL)
            .unwrap();

        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        let tx = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient,
                LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );

        let response = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![encode_transaction(
                        &tx,
                        TransactionBinaryEncoding::Base58,
                    )],
                },
                Some(RpcSimulateBundleConfig {
                    transaction_encoding: Some(UiTransactionEncoding::Base58),
                }),
            )
            .await
            .unwrap();

        assert_eq!(
            response.value.summary,
            RpcSimulateBundleSummary::Succeeded("succeeded".to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_success_base64_default_config() {
        let payer = Keypair::new();
        let recipient = Pubkey::new_unique();
        let setup = TestSetup::new(SurfpoolJitoRpc);

        let _ = setup
            .context
            .svm_locker
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL)
            .unwrap();

        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        let tx = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient,
                LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );

        let response = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![encode_transaction(
                        &tx,
                        TransactionBinaryEncoding::Base64,
                    )],
                },
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            response.value.summary,
            RpcSimulateBundleSummary::Succeeded("succeeded".to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_failure_returns_second_signature() {
        let payer = Keypair::new();
        let recipient = Pubkey::new_unique();
        let setup = TestSetup::new(SurfpoolJitoRpc);

        let _ = setup
            .context
            .svm_locker
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL)
            .unwrap();

        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        let tx1 = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(&payer.pubkey(), &recipient, 1)],
            &recent_blockhash,
        );

        let tx2 = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(
                &payer.pubkey(),
                &recipient,
                100 * LAMPORTS_PER_SOL,
            )],
            &recent_blockhash,
        );

        let response = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![
                        encode_transaction(&tx1, TransactionBinaryEncoding::Base58),
                        encode_transaction(&tx2, TransactionBinaryEncoding::Base58),
                    ],
                },
                Some(RpcSimulateBundleConfig {
                    transaction_encoding: Some(UiTransactionEncoding::Base58),
                }),
            )
            .await
            .unwrap();

        match response.value.summary {
            RpcSimulateBundleSummary::Failed(failed_summary) => {
                assert_eq!(
                    failed_summary.failed.tx_signature,
                    Some(tx2.signatures[0].to_string())
                );
            }
            _ => panic!("expected failed summary"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_is_stateful_across_transactions() {
        let payer = Keypair::new();
        let funded_account = Keypair::new();
        let recipient = Pubkey::new_unique();
        let setup = TestSetup::new(SurfpoolJitoRpc);

        let _ = setup
            .context
            .svm_locker
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL)
            .unwrap();

        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        // tx1 creates and funds a new account.
        let tx1 = build_v0_transaction(
            &payer.pubkey(),
            &[&payer, &funded_account],
            &[system_instruction::create_account(
                &payer.pubkey(),
                &funded_account.pubkey(),
                LAMPORTS_PER_SOL,
                0,
                &system_program::id(),
            )],
            &recent_blockhash,
        );

        // tx2 spends from the newly created account. This only succeeds if tx1 effects are visible.
        let tx2 = build_v0_transaction(
            &funded_account.pubkey(),
            &[&funded_account],
            &[system_instruction::transfer(
                &funded_account.pubkey(),
                &recipient,
                LAMPORTS_PER_SOL / 2,
            )],
            &recent_blockhash,
        );

        let response = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![
                        encode_transaction(&tx1, TransactionBinaryEncoding::Base58),
                        encode_transaction(&tx2, TransactionBinaryEncoding::Base58),
                    ],
                },
                Some(RpcSimulateBundleConfig {
                    transaction_encoding: Some(UiTransactionEncoding::Base58),
                }),
            )
            .await
            .unwrap();

        assert_eq!(
            response.value.summary,
            RpcSimulateBundleSummary::Succeeded("succeeded".to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simulate_bundle_unsupported_encoding_rejected() {
        let payer = Keypair::new();
        let recipient = Pubkey::new_unique();
        let setup = TestSetup::new(SurfpoolJitoRpc);

        let _ = setup
            .context
            .svm_locker
            .airdrop(&payer.pubkey(), 2 * LAMPORTS_PER_SOL)
            .unwrap();

        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|svm_reader| svm_reader.latest_blockhash());

        let tx = build_v0_transaction(
            &payer.pubkey(),
            &[&payer],
            &[system_instruction::transfer(&payer.pubkey(), &recipient, 1)],
            &recent_blockhash,
        );

        let result = setup
            .rpc
            .simulate_bundle(
                Some(setup.context),
                RpcSimulateBundleParams {
                    encoded_transactions: vec![encode_transaction(
                        &tx,
                        TransactionBinaryEncoding::Base58,
                    )],
                },
                Some(RpcSimulateBundleConfig {
                    transaction_encoding: Some(UiTransactionEncoding::JsonParsed),
                }),
            )
            .await;

        let err = result.unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(err.message.contains("unsupported transactionEncoding"));
    }
}
