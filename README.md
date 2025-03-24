본 repository는 한국항공대 AI융합대학의 2025년 PyTorch 코딩 강의자료입니다. 
본 강의는 PyTorch 활용법에 대한 것이며, 딥러닝 이론에 대한 강의 영상은 한국항공대 AI융합대학 유튜브 채널(https://www.youtube.com/watch?v=WQF0KGHm76I&list=PLNxPUUOv7SK7eWZQOSzwHzKsM90sezMsG) 에서 시청하실 수 있습니다.

2025.03.24
Multiple-node 환경일 때, FSDP 실습 코드의 checkpoint save/load 부분이 제대로 돌아가지 않습니다.
코드를 여러 형태로 수정도 해보고, SSH file system을 통해 여러 노드가 공유하는 폴더에 checkpoint를 저장하는 방법도 시도해 보았으나 여전히 해결하지 못한 상태입니다.
현재 PyTorch에서 공개한 자료로는 이 문제를 제대로 해결할 수 없어, 일단은 코드를 현재 그대로 공개하겠습니다.
