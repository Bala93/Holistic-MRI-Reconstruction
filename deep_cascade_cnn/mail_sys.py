import smtplib

def send_mail(exp_info,metric):

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587)
    
    # start TLS for security 
    s.starttls()

    message = '''Subject:{}

    {}'''.format(exp_info,metric)


    # Authentication 
    s.login("balaexperiments@gmail.com", "Health#123")
    
    # sending the mail 
    s.sendmail("balaexperiments@gmail.com", "balaexperiments@gmail.com", message)
    
    # terminating the session 
    s.quit()

    return 


send_mail('Dc-CNN','HFN = 0.333 +/- 0.1368 MSE = 0.0005695 +/- 0.0004549 NMSE = 0.004795 +/- 0.003832 PSNR = 32.75 +/- 3.286 SSIM = 0.9195 +/- 0.0425')
