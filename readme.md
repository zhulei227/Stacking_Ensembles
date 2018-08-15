基于sklearn,keras的stacking ensemble实现,主要功能:

1.继承Classifier类,可自己扩展基分类器,比如classifier_examples.py中的SimpleMLPClassifer类

2.对分类器的CV训练包装:

3.堆叠任意深结构的stacking 分类器

相关demo查看*_examples.py文件