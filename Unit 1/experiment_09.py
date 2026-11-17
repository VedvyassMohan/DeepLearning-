# Assumed example values: prevalence=1%, sensitivity=95%, specificity=90%
prevalence, sensitivity, specificity = 0.01, 0.95, 0.90
false_positive = 1 - specificity
p_positive = sensitivity*prevalence + false_positive*(1-prevalence)
posterior = sensitivity*prevalence / p_positive
print('P(positive test):', round(p_positive, 4))
print('P(disease | positive):', round(posterior, 4))
