본 repository는 한국항공대 AI융합대학의 2025년 PyTorch 코딩 강의자료입니다. 
본 강의는 PyTorch 활용법에 대한 것이며, 딥러닝 이론에 대한 강의 영상은 한국항공대 AI융합대학 유튜브 채널(https://www.youtube.com/watch?v=WQF0KGHm76I&list=PLNxPUUOv7SK7eWZQOSzwHzKsM90sezMsG) 에서 시청하실 수 있습니다.

2025.02.23
FSDP 실습 코드가 checkpoint save/load 부분에 문제가 있어 multiple node 환경에서 제대로 돌아가지 않습니다.
현재 PyTorch에서 공개한 자료로는 이 문제를 제대로 해결할 수 없어, 일단은 코드를 현재 그대로 공개하겠습니다.
이후 FSDP의 full state checkpoint를 대표 node에 저장했다가, 이를 다시 sharded state로 나누어 복원하는 방법을 알게되는대로 코드를 수정하겠습니다.
