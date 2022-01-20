from implementation import visualize_results_runs, visualize_results_runs_2
import os 

def show_01():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/contrastive_learning/results/resnet18' 
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

def show_02():
	results_dir = '/Volumes/cardio-project/cardio/SPUM/CVD_detection_code/contrastive_learning/results/resnet18_apply' 
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

if __name__ == '__main__' : 
	show_01()
	show_02()