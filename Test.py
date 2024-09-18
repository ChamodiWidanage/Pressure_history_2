import tensorflow as tf
#from tensorflow.keras.models import load_model
#ann_arr = tf.keras.models.load_model('RP_Arrival_time/Arrival_ANN4')
#ann_arr.save('files/Arrival_ANN4.h5')

from keras.models import load_model
ann_arr = load_model('RP_Arrival_time/Arrival_ANN4')
ann_end = load_model('RP_End_time/End_ANN4')
ann_chg = load_model('RP_Change_time/Change_ANN1')
#%%
# Save it in the .h5 format
ann_arr.save('files/Arrival_ANN4.h5')
ann_end.save('files/End_ANN4.h5')
ann_end.save('files/Change_ANN1.h5')
#%%
ann1 = load_model('RP_Section_1_new3/Section1_new3_ANN_2')

ann2 = load_model('RP_Section_2_new2/Section2_new_ANN_2')

ann3 = load_model('RP_Section_3_new/Section3_new_ANN_2')

ann4 = load_model('RP_Section_4_new/Section4_new_ANN_7')
#%%
ann1.save('files/Section1_new3_ANN_2.h5')
ann2.save('files/Section2_new_ANN_2.h5')
ann3.save('files/Section3_new_ANN_2.h5')
ann4.save('files/Section4_new_ANN_7.h5')