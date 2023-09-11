import re
import pandas as pd
data = """
188.121.41.140 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.0.2; http://www.cakefantasia.com"
81.169.144.135 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6.1; http://aifs2.pvdveen.net"
91.184.18.50 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://sociedia.com"
184.154.224.17 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6; http://photekusa.com"
209.15.245.58 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://enabledkids.ca"
66.96.183.15 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6; http://guyana-tourism.com/blog"
70.32.98.112 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6.1; http://demo.gavick.com/wordpress/fest"
208.117.12.162 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/2.9.2; http://www.caudillsprouting.com/Latest-News"
184.168.193.106 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://prestogolf.ca/wordpress"
76.163.252.93 - - [23/Sep/2013:13:02:47 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://thevoterupdate.com/jones"
67.192.46.11 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5; http://analytics-lab.tessella.com"
216.201.128.7 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5; http://www.theleadernews.com"
173.201.183.168 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://clonador.net/blog"
209.235.204.197 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://www.imsglobal.org/blog"
50.97.97.131 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6.1; http://awaminationalparty.org/main"
31.22.4.57 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://harvestmoonparadise.com/wp2"
64.111.126.29 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6.1; http://3aocubo.net"
67.43.13.244 - - [23/Sep/2013:13:02:48 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.6; http://blog.aristadba.com"
108.175.6.57 - - [23/Sep/2013:13:02:49 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://spcitytimes.com"
50.56.33.56 - - [23/Sep/2013:13:02:49 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://50.56.33.56/blog"
192.185.4.20 - - [23/Sep/2013:13:02:49 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.4.2; http://www.dhbrownsports.com"
65.98.110.82 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.1.3; http://davasobel.com/blog"
74.52.155.18 - - [23/Sep/2013:13:02:49 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.2; http://mixedmoods.co.uk"
64.90.55.213 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://www.steppir.com"
66.33.212.115 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://www.dsafire.net"
97.74.24.44 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.0; http://www.anarchylane.com/ass/skat"
108.59.11.107 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://pixelsea.biz"
50.116.58.175 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.5.1; http://readydock.net"
97.74.144.179 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.0; http://kentscountrycookies.com/blog"
80.82.113.174 - - [23/Sep/2013:13:02:50 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.0; http://www.davidcoxon.com/run"
74.126.5.225 - - [23/Sep/2013:13:02:51 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/2.9.2; http://www.amray.com/blog"
66.185.31.230 - - [23/Sep/2013:13:02:51 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.3; http://berkeleyreporter.com"
89.151.72.46 - - [23/Sep/2013:13:02:51 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.3.2; http://www.bohomoth.com"
216.15.166.170 - - [23/Sep/2013:13:02:51 +0200] "GET / HTTP/1.1" 200 8955 "-" "WordPress/3.3.2; http://evisthrive.com/blog"


"""

import re
import pandas as pd

# Sample log data
log_data = data

# Initialize lists to store extracted data
ips = []
times = []
urls = []
statuses = []

# Regular expression to extract data from log lines
log_pattern = r'([\d.]+) - - \[([^\]]+)\] "(\S+) ([^"]+)" (\d+)'

# Iterate through log lines and extract data
for line in log_data.strip().split('\n'):
    match = re.match(log_pattern, line)
    if match:
        ip, time, command, url, status = match.groups()
        ips.append(ip)
        times.append(time)
        urls.append(re.sub(r'\sHTTP/1.1$', '', url))  # Remove trailing "HTTP/1.1"
        statuses.append(status)

# Create pandas DataFrame
data = {
    'IP': ips,
    'Time': times,
    'URL': urls,
    'Status': statuses
}
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('log_data.csv', index=False)

# Print the first few rows of the DataFrame
print(df.head())
