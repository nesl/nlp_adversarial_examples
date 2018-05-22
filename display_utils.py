from IPython.core.display import display, HTML
import numpy as np

def html_render(x_orig, x_adv):
    x_orig_words = x_orig.split(' ')
    x_adv_words = x_adv.split(' ')
    orig_html = []
    adv_html = []
    # For now, we assume both original and adversarial text have equal lengths.
    assert(len(x_orig_words) == len(x_adv_words))
    for i in range(len(x_orig_words)):
        if x_orig_words[i] == x_adv_words[i]:
            orig_html.append(x_orig_words[i])
            adv_html.append(x_adv_words[i])
        else:
            orig_html.append(format("<b style='color:green'>%s</b>" %x_orig_words[i]))
            adv_html.append(format("<b style='color:red'>%s</b>" %x_adv_words[i]))
    
    orig_html = ' '.join(orig_html)
    adv_html = ' '.join(adv_html)
    return orig_html, adv_html



def visualize_attack(sess, model, dataset, x_orig, x_adv):
    x_len = np.sum(np.sign(x_orig))
    orig_list = list(x_orig[:x_len])
    adv_list = list(x_adv[:x_len])
    orig_pred = model.predict(sess,x_orig[np.newaxis,:])
    adv_pred = model.predict(sess, x_adv[np.newaxis,:])
    orig_txt = dataset.build_text(orig_list)
    adv_txt = dataset.build_text(adv_list)
    orig_html, adv_html = html_render(orig_txt, adv_txt)
    print('Original Prediction = %s. (Confidence = %0.2f) ' %(('Positive' if np.argmax(orig_pred[0]) == 1 else 'Negative'), np.max(orig_pred)*100.0))
    display(HTML(orig_html))
    print('---------  After attack -------------')
    print('New Prediction = %s. (Confidence = %0.2f) ' %(('Positive' if np.argmax(adv_pred[0]) == 1 else 'Negative'), np.max(adv_pred)*100.0))

    display(HTML(adv_html))
    
    
def visualize_attack2(dataset, test_idx, x_orig, x_adv, label):
    
    raw_text = dataset.test_text[test_idx]
    print('RAW TEXT: ')
    display(HTML(raw_text))
    print('-'*20)
    x_len = np.sum(np.sign(x_orig))
    orig_list = list(x_orig[:x_len])
    #orig_pred = model.predict(sess,x_orig[np.newaxis,:])
    #adv_pred = model.predict(sess, x_adv[np.newaxis,:])
    orig_txt = dataset.build_text(orig_list)
    if x_adv is None:
        adv_txt = "FAILED"
    else:
        adv_list = list(x_adv[:x_len])
        adv_txt = dataset.build_text(adv_list)
    orig_html, adv_html = html_render(orig_txt, adv_txt)
    print('Original Prediction = %s.  ' %('Positive' if label == 1 else 'Negative'))
    display(HTML(orig_html))
    print('---------  After attack -------------')
    print('New Prediction = %s.' %('Positive' if label == 0 else 'Negative'))

    display(HTML(adv_html))