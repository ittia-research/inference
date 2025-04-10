Embedding inferencing on TPU with PyTorch XLA.

§  
VISITOR: Why another inferencing server?  
PROJECT: There are a lot of good inferencing project out there. The project is mainly for edge cases in which we can not find a good existing project yet, mainly embedding inferencing on TPU device. We have noticed that vLLM supports TPU and embedding, but we can not make it working. If you know open-source project that works well, let us know.

## Features
- Supported devices: Google TPU
- Supported models: Alibaba-NLP/gte-Qwen2-1.5B-instruct
- API: REST compateble with OpenAI format

## Quick Start
```bash
git pull https://github.com/ittia-research/inference
cd inference
docker compose up -d
docker compose logs -f
```

## Other Docs
[Roadmap](./docs/roadmap.md)

## Acknowledgements
- TPU Research Cloud team at Google
