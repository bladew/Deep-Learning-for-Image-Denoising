import pretrain1, pretrain2, pretrain3, pretrain4
import train.utils


if __name__ == "__main__":
	images = train.utils.read_images()
	p1 = pretrain1.train(images)
	p2 = pretrain2.train(images)
	p3 = pretrain3.train(images)
	p4 = pretrain4.train(images)


