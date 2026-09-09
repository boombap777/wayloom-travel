# Third-party notices

The MIT LICENSE covers this project's own code and synthetic examples, not upstream
model weights. The verified foundation checkpoint is Qwen/Qwen3-1.7B, revision
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, distributed under Apache-2.0 by Alibaba Cloud.
See docs/licenses/QWEN-APACHE-2.0.txt and the official model distribution:
https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE

The task-specific QLoRA adapter and its GGUF conversion are modified artifacts. Their
model bundle retains the upstream Apache-2.0 license and a notice describing the
fine-tuning and conversion; it does not relabel the base model as MIT.

llama.cpp and the optional PyTorch, Transformers, PEFT, bitsandbytes, Accelerate and
safetensors distributions keep their own licenses and notices. They are not vendored
into the source release. No runtime binary, proprietary travel dataset, live booking
inventory or third-party photo is included in the source package.
