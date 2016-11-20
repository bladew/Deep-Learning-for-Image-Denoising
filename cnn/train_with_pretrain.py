import sys
sys.path.insert(0, './pretrain')
import pretrain1, pretrain2, pretrain3, pretrain4
import train_utils


if __name__ == "__main__":
	images = train_utils.read_images()

	pretrain1.train(images)
	restore_path1 = pretrain1.save()
	
	pretrain2.train(images, restore_path1)
	restore_path2 = pretrain2.save()

	pretrain3.train(images, restore_path2)
	restore_path3 = pretrain3.save()
	
	pretrain4.train(images, restore_path3)
	restore_path4 = pretrain4.save()
'''
	p2 = pretrain2.train(images)
	p3 = pretrain3.train(images)
	p4 = pretrain4.train(images)
	'''

