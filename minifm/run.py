from _libfm import train,predict
coef, v_file, bias, m_sum, m_sum_sqr= train('../downsample','sgd',"c",10)
predict('../downsample','sgd',coef, v_file, bias,m_sum,m_sum_sqr,10, "c")