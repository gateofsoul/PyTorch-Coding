본 repository는 한국항공대 AI융합대학의 2025년 PyTorch 코딩 강의자료입니다. 
본 강의는 PyTorch 활용법에 대한 것이며, 딥러닝 이론에 대한 강의 영상은 한국항공대 AI융합대학 유튜브 채널(https://www.youtube.com/watch?v=WQF0KGHm76I&list=PLNxPUUOv7SK7eWZQOSzwHzKsM90sezMsG) 에서 시청하실 수 있습니다.

2025.03.24
Multiple-node 환경일 때, FSDP 실습 코드의 checkpoint save/load 부분이 제대로 돌아가지 않습니다.
코드를 여러 형태로 수정도 해보고, SSH file system을 통해 여러 노드가 공유하는 폴더에 checkpoint를 저장하는 방법도 시도해 보았으나 여전히 해결하지 못한 상태입니다.
현재 PyTorch에서 공개한 자료로는 이 문제를 제대로 해결할 수 없어, 일단은 코드를 현재 그대로 공개하겠습니다.
다만, checkpoint를 제외한 나머지 기능은 multiple-node에서 정상작동하므로 마음껏 활용하셔도 됩니다.

2025.04.04
nn.LayerNormr과 같은 일부 레이어의 경우, weight와 bias atribute가 있음에도 xavier_normal_을 활용할 수 없기 때문에,
예시 코드에서 parameter_initializer method의 내용을 수정했습니다.

2025.04.08
DDP와 FSDP대한 각종 내용이 PyTorch 2.4 이후 버전에 맞게 수정되었습니다.