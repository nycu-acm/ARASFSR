echo 'celebAHQ-64-80' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 80,80 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-96' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 96,96 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-112' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 112,112 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-128' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 128,128 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-256' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 256,256 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-512' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 512,512 --downsampling bic --gpu $3 &&
echo 'celebAHQ-64-1024' &&
python demo_many.py --model $1 --method_backbone $2 --resolution_in 64,64 --resolution_out 1024,1024 --downsampling bic --gpu $3 &&

true
