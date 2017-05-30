from util.data_parser_t import DataParser
from util.data_dumper import DataDumper
from lib.folds_splitter import get_kfold_indices


if __name__ == "__main__":
	DATASET = 'citeulike-t'

	parser = DataParser(DATASET, 0, "keywords")
	ratings = parser.get_ratings_matrix()
	k_folds = 5
	train_indices, test_indices = get_kfold_indices(ratings, k_folds)
	data_dumper = DataDumper(DATASET, 'splits')
	data_dumper.save_matrix(train_indices, 'train_indices')
	data_dumper.save_matrix(test_indices, 'test_indices')

