isdog = isdog.clip(min=0.05, max=0.95)

filenames = batches.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])

subm = np.stack([ids,isdog], axis=1)
subm[:5]

submission_file_name = 'submission1.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')

# from IPython.display import FileLink
# %cd $LESSON_HOME_DIR
# FileLink('data/redux/'+submission_file_name)
