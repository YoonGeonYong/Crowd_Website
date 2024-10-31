# fromstring 에러
fromstring이 deprecated 됨
```
DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
```
solution
```
Replacing 'fromstring' with 'frombuffer'
```

# matplotlib 경고
main thread 밖에서 실행됨
```
UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
```
solution
```py
import matplotlib
matplotlib.use('agg')
```

# GCP, cv2 오류1
lib.so.1 file이 없음 (로컬 시스템은 존재하지만, 도커 컨테이너에는 누락된 종속성)
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
solution
```
conda remove opencv
pip3 install opencv-python-headless
```

# GCP, GLIBCXX_3.4.29 오류
opencv, torch에서 발생, 해당 파일을 가져올 수 없음
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/juchan/miniconda3/envs/crowded/lib/python3.9/site-packages/torch/lib/libtorch_python.so) 
```
solution
```sh
sudo apt install binutils                                                               # strings 사용하기 위함
strings /home/juchan/miniconda3/envs/crowded/lib/libstdc++.so.6 | grep GLIBCXX          # GLIBCXX_3.4.29 존재 확인

sudo rm /lib/x86_64-linux-gnu/libstdc++.so.6
sudo cp /home/juchan/miniconda3/envs/crowded/lib/libstdc++.so.6 /lib/x86_64-linux-gnu   # libstdc++ 복사
```

# GCP, conda 에러
conda, pip 이것저것 만지다가 발생
```
# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "/home/juchan/miniconda3/lib/python3.12/site-packages/conda/exception_handler.py", line 17, in __call__
        return func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
 ...

An unexpected error has occurred. Conda has prepared the above report.
If you suspect this error is being caused by a malfunctioning plugin,
consider using the --no-plugins option to turn off plugins.

Example: conda --no-plugins install <package>

Alternatively, you can set the CONDA_NO_PLUGINS environment variable on
the command line to run the command without plugins enabled.

Example: CONDA_NO_PLUGINS=true conda install <package>

If submitted, this report will be used by core maintainers to improve
future releases of conda.
Would you like conda to send this report to the core maintainers? [y/N]: 
Timeout reached. No report sent.
```
solution
```sh
conda clean -all        # 캐시 삭제
```

# flask run으로 실행시
- 오류 : app.run(host, port 등등)에 입력한 값이 작동하지 않음
- 문제 : flask run은 기본 테스트 서버를 동작시킴
- 해결 : 환경변수 설정
```
export FLASK_APP=app.py
export FLASK_RUN_PORT=8080  # 포트 번호 설정
export FLASK_RUN_HOST=0.0.0.0  # 호스트 설정 (필요한 경우)
flask run
```

# python run.py로 flask app 실행시
- 오류 : 처음 시작시 코드 변경이 없어도, 재로딩이 한번 발생
- 이유 : create_app() 내부에, config.py로 환경설정 하는 부분에서 debug=true 항목이 있음 -> 이를 읽고 변경으로 감지


# model.load_state_dict() 사용시 에러
- 에러
```
raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for MCNN:
        Missing key(s) in state_dict: "branch1.0.conv.weight", "branch1.0.conv.bias", "branch1.2.conv.weight", "branch1.2.conv.bias", "branch1.4.conv.weight", "branch1.4.conv.bias", "branch1.5.conv.weight", "branch1.5.conv.bias", "branch2.0.conv.weight", "branch2.0.conv.bias", "branch2.2.conv.weight", "branch2.2.conv.bias", "branch2.4.conv.weight", "branch2.4.conv.bias", "branch2.5.conv.weight", "branch2.5.conv.bias", "branch3.0.conv.weight", "branch3.0.conv.bias", "branch3.2.conv.weight", "branch3.2.conv.bias", "branch3.4.conv.weight", "branch3.4.conv.bias", "branch3.5.conv.weight", "branch3.5.conv.bias", "fuse.0.conv.weight", "fuse.0.conv.bias". 
        Unexpected key(s) in state_dict: "DME.branch1.0.conv.weight", "DME.branch1.0.conv.bias", "DME.branch1.2.conv.weight", "DME.branch1.2.conv.bias", "DME.branch1.4.conv.weight", "DME.branch1.4.conv.bias", "DME.branch1.5.conv.weight", "DME.branch1.5.conv.bias", "DME.branch2.0.conv.weight", "DME.branch2.0.conv.bias", "DME.branch2.2.conv.weight", "DME.branch2.2.conv.bias", "DME.branch2.4.conv.weight", "DME.branch2.4.conv.bias", "DME.branch2.5.conv.weight", "DME.branch2.5.conv.bias", "DME.branch3.0.conv.weight", "DME.branch3.0.conv.bias", "DME.branch3.2.conv.weight", "DME.branch3.2.conv.bias", "DME.branch3.4.conv.weight", "DME.branch3.4.conv.bias", "DME.branch3.5.conv.weight", "DME.branch3.5.conv.bias", "DME.fuse.0.conv.weight", "DME.fuse.0.conv.bias". 
```
- 이유 : trained_A.pth 파일에 저장된 모델과 다름, layer 이름까지 동일해야 함
```
기존 : DME.conv.weight
현재 : conv.weight
```
- 해결1 : strict=False 옵션 사용 (일부 레이어만 일치할 경우 로드, 하지만 일부 레이어가 제대로 로드되지 않을 수 있음)
```py
model.load_state_dict(torch.load('mcnn/trained_A.pth'), strict=False)
```
- 해결2 : trained_A.pth에 DME. 접두사 제거 후 다시 저장
```py
import torch
from collections import OrderedDict
from mcnn.models import MCNN

cp = torch.load('mcnn/trained_A.pth')

new_state_dict = OrderedDict()
for k, v in cp.items():
    new_k = k.replace('DME.', '')
    new_state_dict[new_k] = v

model = MCNN()
model.load_state_dict(new_state_dict)
torch.save(model.state_dict(), 'trained_A_new.pth')
```

# MCNN 모델에 데이터 입력시 오류
- 에러
```
TypeError: conv2d() received an invalid combination of arguments - got (numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of:
 * (Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups)
      didn't match because some of the arguments have invalid types: (numpy.ndarray, Parameter, Parameter, tuple of (int, int), tuple of (int, int), tuple of (int, int), int)
```
- 이유 : input type이 맞지 않음 (numpy X)
- 해결 : numpy -> tensor
```py
_in = torch.from_numpy(_in).type(torch.FloatTensor).to(torch.device('mps'))
```

# cv applyColorMap 적용시 에러
- 에러
```
cv2.error: OpenCV(4.6.0) /private/var/folders/k1/30mswbxs7r1g6zwn8y4fyt500000gp/T/abs_11nitadzeg/croot/opencv-suite_1691620374638/work/modules/imgproc/src/colormap.cpp:736: error: (-5:Bad argument) cv::ColorMap only supports source images of type CV_8UC1 or CV_8UC3 in function 'operator()'
```
- 문제 : src img 타입이 float이라 문제 발생
- 해결 : uint8으로 변환
```py
img = (img * 255).astype(np.uint8)
```

# canvas 이미지가 표시되지 않는 현상
- 에러 : 이미지가 없고, 웹 서버로 전송되는 이미지가 까맣다
- 문제 : canvas, video 크기가 다름
- 해결 : 크기 통일


# git push 에러
- 에러
```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```
- 문제 : 깃에 ssh키가 설정되지 않음
- 해결 : ssh키 설정 -> 깃 등록
```sh
ssh-keygen -t rsa -C leejuchan317@gmail.com  # ssh키 생성 (기본 위치: ~/.ssh/id_rsa.pub)  
cat ~/.ssh/id_rsa.pub # ssh키 확인 -> 깃에 복사

# Settings -> SSH and GPS Keys -> New SSH key -> 복사
```
