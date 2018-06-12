import numpy as np
import pickle
import glob
import os
from collections import Counter
from PIL import Image
import argparse

def calculatePQ(pred, label, num_class, pq_log):
	pred_class = pred // 1000
	pred_instance = pred % 1000
	label_class = label // 1000
	label_instance = label % 1000
	class_list = ["0: back ground"]
	pq_dict = {0: -1}

	for i in range(1, num_class):
		bool_pred_class = pred_class == i	# クラスiのとこだけTrue
		bool_label_class = label_class == i

		if (np.sum(bool_label_class)) == 0:
			class_list.append(str(i) + ": not exist")
			pq_dict[i] = -1
			continue

		IoU = 0.
		tp = fp = fn = 0
		correspond_instance = {}
		label_instance_list = np.reshape(label_instance[bool_label_class], [-1]).tolist()
		label_instance_counter = Counter(label_instance_list)

		for id, _ in sorted(label_instance_counter.items(), key=lambda x: -x[1]):
			bool_label_instance = ((label_instance == id) * bool_label_class)
			p_instance = pred_instance[bool_pred_class]
			p_instance_list = list(filter(lambda  x: x not in correspond_instance.values(), np.reshape(p_instance, [-1]).tolist()))
			if (len(p_instance_list) == 0):
				continue
			p_instance_counter = Counter(p_instance_list)
			max_p_instance = max(p_instance_counter.items(), key=lambda x:x[1])[0]
			correspond_instance[id]=(max_p_instance)
			bool_pred_instance = (pred_instance == max_p_instance) * bool_pred_class
			temp_tp = np.sum(np.logical_and(bool_label_instance, bool_pred_instance), dtype=np.uint32)
			temp_fp = np.sum(np.logical_and(np.logical_not(bool_label_instance), bool_pred_instance), dtype=np.uint32)
			temp_fn = np.sum(np.logical_and(bool_label_instance, np.logical_not(bool_pred_instance)), dtype=np.uint32)
			if (temp_tp > 0) :
				tp += 1
			else:
				continue
			if (temp_fp > 0) :
				fp_unique = np.unique(label_instance[bool_pred_instance * bool_label_class])
				len_fp = len(fp_unique)
				#print("fp", id, fp_unique, np.unique(label_instance[bool_label_class]))
				fp += len_fp - 1 if len_fp else 0
			if (temp_fn > 0) :
				fn_unique = np.unique(pred_instance[bool_label_instance * bool_pred_class])
				len_fn = len(fn_unique)
				#print("fn", max_p_instance, fn_unique, np.unique(pred_instance[bool_pred_class]))
				fn += len_fn - 1 if len_fn else 0
			#t_IoU =temp_tp / (temp_tp + temp_fn + temp_fp)
			IoU += temp_tp / (temp_tp + temp_fn + temp_fp)
		if (tp > 0):
			#print('class', i, 'IoU', IoU, 'tp', tp,'fp', fp,'fn', fn )
			a = [str(i[0]) + ': ' + str(i[1]) for i in list(correspond_instance.items())]
			class_list.append(str(i) + ": corresponded_instance={" + ', '.join(a) + "} , (tp, fp, fn)=("+str(tp)+","+str(fp)+","+str(fn)+")")
			#class_list.append(str(i) + ": used_p_inst" + str(used_pred_instance))
			pq_dict[i] = (IoU / tp, tp / (tp + 0.5 * fp + 0.5 * fn))
		else:
			class_list.append(str(i) + ": tp=0")
			pq_dict[i] = -1

	print("unique_label_class", np.unique(label_class).tolist())
	print("unique_pred_class ", np.unique(pred_class).tolist())
	print("class_list", class_list)
	pq_log.append(','.join(class_list))
	print(pq_dict)
	a = [ str(i[0])+':'+str(i[1]) for i in list(pq_dict.items()) ]
	pq_log.append(', '.join(a))
	pq = class_count = 0
	for i in pq_dict.items():
		if i[1] != -1 :
			pq += i[1][0] * i[1][1]
			class_count += 1
	if(class_count):
		print('pq=', pq / class_count, ', pq_sum', pq , ', num_class', class_count)
		pq_log.append('pq=' + str(pq / class_count) + ', pq_sum'+str(pq) + ', num_class' + str(class_count))
	else:
		print("--- Prediction for all class is wrong. ---")
		pq_log.append("--- Prediction for all class is wrong. ---")

'''
pred = np.array([
	[	   0, 1000, 1000	],
	[	2000, 2000, 2000	]
	#[	3000, 3000, 3001, 3001, 3002	]
])
label = np.array([
	[	   0, 1001, 1001	],
	[	2001, 2001, 2000	]
	#[3000, 3000, 3001, 3000, 3002]
])

calculatePQ(pred, label, 4)
print("end")
ユーザー: 自炊にそこまで興味がない学生 は、
欲求: おいしいものを食べたいが自炊につかう時間を減らしたい が、
課題: 時間を減らす技術や知識がない ので、
製品の特徴: 時短クッキング に価値がある
検証方法: ___学生を呼んで時短クッキングをさせてみ_____
'''
if __name__ == '__main__':
	usage = '%(prog)s <pred_dir> <label_dir> <log.txt>'
	parser = argparse.ArgumentParser(description='calculate PQ between pickles in <pred_dir>-directory and png-images in <label_dir>-directory and save the log in <log.txt>', usage = usage)
	parser.add_argument("convert", type=str, nargs=3, help='look usage. %(prog)s requires three argumemts.')
	args = parser.parse_args()

	#PRED_DIR = '/media/deep/sekine/DeepDrive/result/10_class_mask-epoch30_seg-futher2-epoch990_16bit/id/'
	#LABEL_DIR = '/media/deep/sekine/DeepDrive/val/DS_id_v2_serial/'
	PRED_DIR = args.convert[0]
	LABEL_DIR = args.convert[1]
	pred_names = glob.glob(PRED_DIR + '*')
	label_names = glob.glob(LABEL_DIR + '*')
	NUM_CLASS = 21
	count = 0
	pq_log = []

	for label_name in label_names:
		label_name = os.path.basename(label_name)

		#if (count > 5):
		#	break
		print(count, label_name)
		pq_log.append(str(count) + ' : ' +  label_name)
		count += 1

		with open(LABEL_DIR + label_name, "rb") as f:
			l = pickle.load(f)
		l_name, ext = os.path.splitext(label_name)
		if(PRED_DIR + l_name + '.pickle' in pred_names):
			with open(PRED_DIR + l_name + '.pickle', "rb") as f:
				p = pickle.load(f)
		elif (PRED_DIR + l_name + '.png' in pred_names):
			p_img = Image.open(PRED_DIR + l_name + '.png')
			p = np.asarray(p_img)
		else:
			print("not found")
			continue
		l = np.reshape(l, [-1])
		p = np.reshape(p, [-1])
		calculatePQ(p, l, NUM_CLASS, pq_log)

	with open(args.convert[2], 'w') as f:
		for l in pq_log:
			f.write(l +'\n')
	print("end")
	#'''
