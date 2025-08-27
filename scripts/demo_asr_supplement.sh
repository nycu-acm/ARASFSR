echo 'celebAHQ-32-128' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 32,32 --resolution_out 128,128 --downsampling bic --gpu $3 &&
echo 'celebAHQ-32-256' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 32,32 --resolution_out 256,256 --downsampling bic --gpu $3 &&
echo 'celebAHQ-32-512' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 32,32 --resolution_out 512,512 --downsampling bic --gpu $3 &&
echo 'celebAHQ-96-256' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 96,96 --resolution_out 256,256 --downsampling bic --gpu $3 &&
echo 'celebAHQ-96-512' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 96,96 --resolution_out 512,512 --downsampling bic --gpu $3 &&
echo 'celebAHQ-96-1024' &&
python demo_many_only_test.py --model $1 --method_backbone $2 --resolution_in 96,96 --resolution_out 1024,1024 --downsampling bic --gpu $3 &&

true
