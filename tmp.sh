# TARGET=HTvjzsfX3yU6BUodCjZ5vZkUrAxMDTrBs3CJaq43ashR
TARGET=8N2mFnZRnTGtKrZPKQ2tp6MhCqqy1P5i28Y5HuM8CeMd
SENDER=/tmp/sender-42.json

# solana-keygen new --no-bip39-passphrase --silent --force -o "$SENDER"
SENDER_PUB=$(solana-keygen pubkey "$SENDER")

# fund sender in Surfpool
curl -s -X POST http://127.0.0.1:8899 -H 'content-type: application/json' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"surfnet_setAccount\",\"params\":[\"$SENDER_PUB\",{\"lamports\":2000000000}]}" | jq

# terminal A: subscribe (auto-exits after 20s)
# grpcurl -plaintext -import-path /tmp -proto geyser.proto \
#   -d "{\"transactions\":{\"watch\":{\"vote\":false,\"failed\":false,\"account_include\":[\"$TARGET\"]}},\"commitment\":\"PROCESSED\"}" \
#   127.0.0.1:2503 geyser.Geyser/Subscribe

commitment="PROCESSED"
# commitment="CONFIRMED"
# commitment="FINALIZED"
echo "Subscribing to $TARGET with commitment $commitment"

grpcurl -plaintext -import-path /tmp -proto geyser.proto \
  -d "{\"accounts\":{\"watch\":{\"account\":[\"$TARGET\"]}},\"commitment\":\"$commitment\"}" \
  127.0.0.1:2503 geyser.Geyser/Subscribe
