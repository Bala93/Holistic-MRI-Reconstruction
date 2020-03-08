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
