import numpy as np
import pandas as pd
import sklearn.model_selection



abundances=['Na/Fe', 'O/Fe']
def data_loader(target=abundances, xysplit=False, load_test_data=False, full_data_filename='../Datasets/Original_Datasets/NGC_combi.csv', data_folder='../Datasets'):
#This function loads the CSV file corresponding to a given target abundance
	if target=='FULL':
		df=pd.read_csv(full_data_filename)
		df.set_index('Star ID', inplace=True)
		if 'F435W' in df.columns:
			df.rename(columns={'F435W': 'F438W', 'F435W_abs':'F438W_abs', 'F435W_e':'F438W_e'}, inplace=True)
		return df
	elif target==abundances:
		prefix=''
	else:
		prefix=f"{target.replace('/', '_')}_"
	train=pd.read_csv(f'{data_folder}/{prefix}training_data.csv')
	train.set_index('Star ID', inplace=True)
	if 'F435W' in train.columns:
		train.rename(columns={'F435W': 'F438W', 'F435W_abs':'F438W_abs', 'F435W_e':'F438W_e'}, inplace=True)
	if load_test_data==True:
		test=pd.read_csv(f'{data_folder}/{prefix}TEST_DATA.csv')
		test.set_index('Star ID', inplace=True)
		if 'F435W' in test.columns:
			test.rename(columns={'F435W': 'F438W', 'F435W_abs':'F438W_abs', 'F435W_e':'F438W_e'}, inplace=True)
		if xysplit==True:
			X_train, X_test, y_train, y_test=train.drop(target, axis=1), test.drop(target, axis=1), train[target], test[target]
			return X_train, X_test, y_train, y_test #functions end when a return line is reached, regardless of where it is. So we can use it for "blocking"
		return train, test
	if xysplit==True:
		X_train, y_train=train.drop(target, axis=1), pd.DataFrame({'Star ID': train.index, target:train[target]}).set_index('Star ID')
		return X_train, y_train
	return train

	
def train_test_split(df, test_size, xysplit=False, target=abundances, stratify='NGC', random_state=42):
#This function takes a full dataframe and splits it into a test and training set nicely regardless of group sizes. If given a test size>1, it will calculate the % of the rows from df that it needs to select for testing (recommended). If given a %, it will use that % of df (not recommended).
	#convert test size to % of df rows
	if test_size>1.:
		test_size=np.float64(test_size)/df.shape[0] 
	else:
		print(f'Using {100*test_size} percent of the passed df as test size, not necessarily {100*test_size} percent of the full dataset') 
	membership_threshold=np.ceil(1./test_size)
	population=df[stratify].value_counts()
	lowpop=population[population<membership_threshold]
	small_groups=lowpop.index.tolist()
	stars_in_small_groups=df[df[stratify].isin(small_groups)]
	#need this for different, updated syntax which allows us to control random state of choice() function
	rng = np.random.default_rng(random_state)
	small_test_ids=[]
	for grp in small_groups:
		num_stars=population.loc[grp]
		#Star ID is used as the index; the location index of the star ID is used to look it up from np.where()
		location_indeces=np.where(stars_in_small_groups[stratify]==grp)
    	#If there is only one star in the group, add it to the test set
		if num_stars==1:
			id=stars_in_small_groups.index[location_indeces[0][0]]
		else:
			#value=np.random.choice(stars_in_small_groups.index[location_indeces])
        	#In order to control the random state of this function, need to use the "rng" version below
			id=rng.choice(stars_in_small_groups.index[location_indeces])
		small_test_ids.append(id)	
	small_test=stars_in_small_groups.loc[np.array(small_test_ids)]
	small_train=stars_in_small_groups.drop(np.array(small_test_ids))
	
	#Now we have our correctly split testing and training data from small groups. Need to filter from main dataframe, then split and combine:
	df=df[~df[stratify].isin(small_groups)]
	big_train, big_test=sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify])
	train, test=pd.concat([big_train, small_train]), pd.concat([big_test, small_test])
	
	if xysplit==True:
		X_train, X_test, y_train, y_test=train.drop(target, axis=1), test.drop(target, axis=1), train[target], test[target]
		return X_train, X_test, y_train, y_test
	else:
		return train, test
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
    	
