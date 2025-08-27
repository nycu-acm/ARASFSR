echo 'celebAHQ-12-96' &&
python demo_many_only_test_new.py --model $1 --method_backbone $2 --resolution_in 12,12 --resolution_out 96,96 --downsampling bic --gpu $3 &&
echo 'celebAHQ-16-128' &&
python demo_many_only_test_new.py --model $1 --method_backbone $2 --resolution_in 16,16 --resolution_out 128,128 --downsampling bic --gpu $3 &&
echo 'celebAHQ-20-160' &&
python demo_many_only_test_new.py --model $1 --method_backbone $2 --resolution_in 20,20 --resolution_out 160,160 --downsampling bic --gpu $3 &&
echo 'celebAHQ-32-256' &&
python demo_many_only_test_new.py --model $1 --method_backbone $2 --resolution_in 32,32 --resolution_out 256,256 --downsampling bic --gpu $3 &&

true
