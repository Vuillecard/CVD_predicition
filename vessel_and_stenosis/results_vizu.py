from implementation import visualize_results_runs, visualize_results_runs_2
import os 
import json
import pickle

def show_01():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/resnet18' 
	#names = ['baseline_reg','baseline_']
	names = ['baseline', 'weighted_loss', 'weighted_dataloader', 'augmented_culprit']
	num_fold = 5
	title_plots = [ ' Baseline, patient CV-%d summaries'%(num_fold), ' Weighted loss, patient CV-%d summaries'%(num_fold) ,
					' Balance dataloader, patient CV-%d summaries'%(num_fold) , ' Augmented MI, patient CV-%d summaries'%(num_fold)]
	for name, title_plot in zip(names,title_plots) : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)],
					'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)] , 
					'Fine-tune best': [ name+'_perf_ft_best'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)]}

		results_save = results_dir
		visualize_results_runs( name_files, results_save, os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2( name_files, results_save, os.path.join(results_dir, 'plot_' + name + '2.png'), title_plot )

def show_02():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/test_augmented_culprit' 
	#names = ['baseline_reg','baseline_']
	names = [ 'augmented_culprit']
	for name in names : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_cv_'+str(k+1)+'.pkl' for k in range(3)],
					   'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_cv_'+str(k+1)+'.pkl' for k in range(3)] , 
					   'Fine-tune best': [ name+'_perf_ft_best'+'_cv_'+str(k+1)+'.pkl' for k in range(3)]}

		results_save = results_dir
		visualize_results_runs(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '2.png'))

def show_03():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/siamese' 
	#names = ['baseline_reg','baseline_']
	names = ['baseline', 'weighted_loss', 'weighted_dataloader', 'augmented_culprit']
	num_fold = 5
	title_plots = [ ' Baseline, patient CV-%d summaries'%(num_fold), ' Weighted loss, patient CV-%d summaries'%(num_fold) ,
					' Balance dataloader, patient CV-%d summaries'%(num_fold) , ' Augmented MI, patient CV-%d summaries'%(num_fold)]
	for name,title in zip(names,title_plots) : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)],
					'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)] , 
					'Fine-tune best': [ name+'_perf_ft_best'+'_cv_'+str(k+1)+'.pkl' for k in range(num_fold)]}

		results_save = results_dir
		visualize_results_runs(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '2.png'),title)

def show_04():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/stenosis' 
	names = ['baseline','augmentation','aug_reg']
	#names = ['augmentation','aug_reg']
	for name in names : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_cv_'+str(k+1)+'.pkl' for k in range(2)] ,
						'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_cv_'+str(k+1)+'.pkl' for k in range(2)] , 
						'Fine-tune best': [ name+'_perf_ft_best'+'_cv_'+str(k+1)+'.pkl' for k in range(3)]}

		results_save = results_dir
		visualize_results_runs(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '2.png'),'todo')

def show_06():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results' 
	names = ['baseline','augmentation']
	name_results= [['tuning_base_stenosis_2_scratch','tuning_base_stenosis_2','tuning_base_stenosis_2_best'],
						 ['tuning_aug_stenosis_scratch','tuning_aug_stenosis','tuning_aug_stenosis_best']]
	exp_results= [['exp_9','exp_9','exp_14'],
				  ['exp_13','exp_5','exp_9']]
	config = ['Scratch','Fine-tune imgNet','Fine-tune best']
	#names = ['augmentation','aug_reg']
	for name_i,name in enumerate(names) :
		name_files = {}
		for i in range(3):
			file_config_path= os.path.join(results_dir,name_results[name_i][i],'config_prameter_tuning.json')
			file_config = json.load( open( file_config_path ))
           
			name_files[config[i]] =  [ os.path.join(results_dir,name_results[name_i][i],file) for file in file_config[exp_results[name_i][i]]['paths']  ]
						

		results_save = results_dir
		visualize_results_runs(name_files,None,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,None,os.path.join(results_dir, 'plot_' + name + '2.png'),'todo')

def show_07():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results' 
	names = ['baseline','augmentation']
	name_results= [['tuning_base_stenosis_2_scratch','tuning_base_stenosis_2','tuning_base_stenosis_2_best'],
						 ['tuning_aug_stenosis_scratch','tuning_aug_stenosis','tuning_aug_stenosis_best']]
	# Best experiment based on the accuracy
	exp_results= [['exp_11','exp_8','exp_16'],
				  ['exp_16','exp_2','exp_2']]
	config = ['Scratch','Fine-tune imgNet','Fine-tune best']
	title = ['Baseline, patient CV-5 summaries','Augmentation, patient CV-5 summaries']
	#names = ['augmentation','aug_reg']
	for name_i,name in enumerate(names) :
		name_files = {}
		for i in range(3):
			file_config_path= os.path.join(results_dir,name_results[name_i][i],'config_prameter_tuning.json')
			file_config = json.load( open( file_config_path ))
           
			name_files[config[i]] =  [ os.path.join(results_dir,name_results[name_i][i],file) for file in file_config[exp_results[name_i][i]]['paths']  ]
						
		results_save = results_dir
		visualize_results_runs(name_files,None,os.path.join(results_dir,'stenosis', 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,None,os.path.join(results_dir,'stenosis', 'plot_' + name + '2.png'),title[name_i])

def show_05():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/testing' 
	names = ['augmented_culprit']
	#names = ['augmentation','aug_reg']
	for name in names : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_run_'+str(k+1)+'.pkl' for k in range(5)] ,
						'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_run_'+str(k+1)+'.pkl' for k in range(5)] , 
						'Fine-tune best': [ name+'_perf_ft_best'+'_run_'+str(k+1)+'.pkl' for k in range(5)]}

		results_save = results_dir
		visualize_results_runs(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '2.png'),'testing augmented culprit')

def show_08():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/contrastive_learning/results' 
	names = ['baseline']
	name_results= [['tuning_balance_apply','tuning_balance_channel']]
	# Best experiment based on the accuracy
	exp_results= [['exp_16','exp_15']]
	config = ['Apply','Channel']
	title = ['Best configuration with ImageNet initialization, patient CV-4 summaries']
	#names = ['augmentation','aug_reg']
	for name_i,name in enumerate(names) :
		name_files = {}
		for i in range(2):
			file_config_path= os.path.join(results_dir,name_results[name_i][i],'config_prameter_tuning.json')
			file_config = json.load( open( file_config_path ))
           
			name_files[config[i]] =  [ os.path.join(results_dir,name_results[name_i][i],file) for file in file_config[exp_results[name_i][i]]['paths']  ]
						
		results_save = results_dir
		visualize_results_runs(name_files,None,os.path.join(results_dir,'plot_tuning', 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,None,os.path.join(results_dir,'plot_tuning', 'plot_' + name + '2.png'),title[name_i])

def show_09():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/cnn/results/stenosis_baseline_test' 
	names = ['baseline']
	#names = ['augmentation','aug_reg']
	for name in names : 
		name_files = { 'Scratch': [ name+'_perf_scratch'+'_run_'+str(k+1)+'.pkl' for k in range(5)] ,
						'Fine-tune imgNet':[ name+'_perf_ft_imgnet'+'_run_'+str(k+1)+'.pkl' for k in range(5)] , 
						'Fine-tune best': [ name+'_perf_ft_best'+'_run_'+str(k+1)+'.pkl' for k in range(5)]}

		results_save = results_dir
		visualize_results_runs(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '.png'))
		visualize_results_runs_2(name_files,results_save,os.path.join(results_dir, 'plot_' + name + '2.png'),'testing augmented culprit')
if __name__ == '__main__':
	#show_01()
	#show_03()
	
	#show_05()
	#show_04()
	#show_07()
	#show_08()
	show_09()

