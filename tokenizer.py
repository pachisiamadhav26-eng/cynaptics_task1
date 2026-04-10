from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder       = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size     = 3000,
    special_tokens = ["[UNK]"]
)

tokenizer.train(["shakespeare.txt"], trainer)
tokenizer.save("shakespeare-tokenizer-v2.json")

# verify
vocab_size = tokenizer.get_vocab_size()
print(f"vocab_size: {vocab_size}")

# test
output = tokenizer.encode("Being an agreed cast")
print(f"tokens : {output.tokens}")
print(f"decoded: {tokenizer.decode(output.ids)}")









