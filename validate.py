import numpy as np

# pred - predicted labels
# b_index - total amount of building points in ref if building
def calculate_stats(pred, b_index):

	_, temp = np.unique(pred[:b_index] == 1, return_counts=True)
	FN, TP = temp[0], temp[1]
	_, temp = np.unique(pred[b_index:] == 1, return_counts=True)
	TN, FP = temp[0], temp[1]

	accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
	recall = TP / (TP + FN) * 100
	specificity = TN / (TN + FP) * 100
	precision = TP / (TP + FP) * 100

	return TP, FP, FN, TN, accuracy, recall, specificity, precision


if __name__ == '__main__':
	pred = np.loadtxt('models/vecpilseta/17_02_2022-14_15/predicted_region.txt')
	ref_b = np.loadtxt('data/vecpilseta/extracted_points/building.txt')
	ref_v = np.loadtxt('data/vecpilseta/extracted_points/vegetation.txt')

	ref_labels = np.zeros((ref_b.shape[0] + ref_v.shape[0]))
	ref_labels[:ref_b.shape[0]] = 1

	TP, FP, FN, TN, accuracy, recall, specificity, precision = calculate_stats(pred[:, 3], ref_b.shape[0])
	print(TP, FP, FN, TN, accuracy, recall, specificity, precision)