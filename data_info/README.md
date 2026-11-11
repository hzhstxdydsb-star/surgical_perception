# 数据准备与类别

采用CholecInstanceSeg官方图像、实例JSON与元数据，来源：https://github.com/cai4cai/cholec_instance_seg 。

下载文件保存到datasets/cholecinstanceseg/raw/：cholecinstance_seg_reference_image_set.zip、cholecinstanceseg.zip、cholecinstanceseg_metadata.csv。图像压缩包解压为cholecinstance_seg_reference_image_set/，保留split/sequence/img_dir/结构。

运行scripts/prepare_data.py后，在datasets/cholecinstanceseg/prepared/生成图像、YOLO标签和data.yaml；独立检查使用scripts/verify_prepared.py。scripts/run.py会根据data.yaml所在位置解析数据根目录。

训练26,830帧、验证4,284帧、测试10,819帧，共64,755个实例。data_info/splits/的文件名相对于prepared目录。类别编号为0 grasper、1 hook、2 irrigator、3 clipper、4 bipolar、5 scissors、6 snare。

完整数据保留在本地项目datasets/cholecinstanceseg/，完整转换记录保留在原始实验归档data/conversion.jsonl。此目录提供划分清单与核查摘要；报告示例是固定抽样的20张验证图像。
