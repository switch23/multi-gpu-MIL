# multi-gpu-MIL

PyTorchのマルチプロセス化機能によってマルチインスタンス学習(MIL)をGPU並列化する

# 使い方

OriginalDataset.pyの`DATA_PATH = f'data_directry'`を自分のデータセットのpathに変更し, dataset.pyを使用するデータセットに変更 
MIL_train.shの設定を自分の環境に合わせて`qsub MIL_train.sh`で実行可能

