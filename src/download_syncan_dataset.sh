# Downloading syncan dataset from the github repo
git clone https://github.com/etas/SynCAN.git ../../datasets/can-ids/syncan/
echo "Raw SynCAN dataset downloaded in ../../datasets/can-ids/syncan/"
cd ../../datasets/can-ids/syncan/
unzip 'train_*.zip' -d ambients
echo "Unzipped training dataset in datasets/can-ids/syncan/ambients"
unzip 'test_*.zip' -d attacks
echo "Unzipped training dataset in datasets/can-ids/syncan/attacks"
rm -rf *.zip
rm -rf attacks/test_normal*
echo "SyncCAN Data Downloaded!"
