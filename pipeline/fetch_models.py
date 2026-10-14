"""Download the two pretrained models at their pinned revisions into $HF_HOME (mandatory setup step, README "Setup").
Local: HF_HOME=$BER_SCRATCH/hf.  Node: cloud/env_setup.sh runs this with HF_HOME=$HOME/hf (the path every cloud/ launcher uses).
  python pipeline/fetch_models.py"""
import os, sys
from huggingface_hub import snapshot_download

PINS = {'intfloat/multilingual-e5-base': 'd128750597153bb5987e10b1c3493a34e5a4502a',    # start of every cross-encoder
        'intfloat/multilingual-e5-small': '614241f622f53c4eeff9890bdc4f31cfecc418b3'}   # dense retrieval embeddings
FILES = ['*.json', '*.safetensors', 'sentencepiece.bpe.model', '1_Pooling/*']
hf = os.environ.get('HF_HOME')
if not hf:
    sys.exit('set HF_HOME first (local: $BER_SCRATCH/hf; node: $HOME/hf)')
for repo, rev in PINS.items():
    p = snapshot_download(repo, revision=rev, allow_patterns=FILES)
    exp = os.path.join(hf, 'hub', 'models--' + repo.replace('/', '--'), 'snapshots', rev)
    assert os.path.realpath(p) == os.path.realpath(exp), (p, exp)
    assert os.path.isfile(os.path.join(p, 'model.safetensors')) and os.path.isfile(os.path.join(p, 'tokenizer.json')), p
    print(f'{repo} @ {rev} -> {p}')
