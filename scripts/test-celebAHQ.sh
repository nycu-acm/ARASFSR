echo 'celebAHQ-16-24' &&
python test.py --config ./configs/test/test-celebAHQ-16-24.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-16-32' &&
python test.py --config ./configs/test/test-celebAHQ-16-32.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-16-64' &&
python test.py --config ./configs/test/test-celebAHQ-16-64.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-16-128' &&
python test.py --config ./configs/test/test-celebAHQ-16-128.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-16-256' &&
python test.py --config ./configs/test/test-celebAHQ-16-256.yaml --model $1 --gpu $2 &&

echo 'celebAHQ-32-48' &&
python test.py --config ./configs/test/test-celebAHQ-32-48.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-32-64' &&
python test.py --config ./configs/test/test-celebAHQ-32-64.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-32-128' &&
python test.py --config ./configs/test/test-celebAHQ-32-128.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-32-256' &&
python test.py --config ./configs/test/test-celebAHQ-32-256.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-32-512' &&
python test.py --config ./configs/test/test-celebAHQ-32-512.yaml --model $1 --gpu $2 &&

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

echo 'celebAHQ-128-192' &&
python test.py --config ./configs/test/test-celebAHQ-128-192.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-128-256' &&
python test.py --config ./configs/test/test-celebAHQ-128-256.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-128-512' &&
python test.py --config ./configs/test/test-celebAHQ-128-512.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-128-1024' &&
python test.py --config ./configs/test/test-celebAHQ-128-1024.yaml --model $1 --gpu $2 &&

echo 'celebAHQ-256-384' &&
python test.py --config ./configs/test/test-celebAHQ-256-384.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-256-512' &&
python test.py --config ./configs/test/test-celebAHQ-256-512.yaml --model $1 --gpu $2 &&
echo 'celebAHQ-256-1024' &&
python test.py --config ./configs/test/test-celebAHQ-256-1024.yaml --model $1 --gpu $2 &&

true
