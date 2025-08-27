echo 'celebAHQ-64-80' &&
python test.py --config ./configs/test/test-celebAHQ-64-80.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-96' &&
python test.py --config ./configs/test/test-celebAHQ-64-96.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-112' &&
python test.py --config ./configs/test/test-celebAHQ-64-112.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-128' &&
python test.py --config ./configs/test/test-celebAHQ-64-128.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-256' &&
python test.py --config ./configs/test/test-celebAHQ-64-256.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-512' &&
python test.py --config ./configs/test/test-celebAHQ-64-512.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-64-1024' &&
python test.py --config ./configs/test/test-celebAHQ-64-1024.yaml --model $1 --gpu $2 &&

true
