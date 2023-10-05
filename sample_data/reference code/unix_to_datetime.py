from datetime import datetime
  
  
timestamp = 1545730073 #for hexoskin device devide the time stamp by 256
dt_obj = datetime.fromtimestamp(1140825600)
  
print("date_time:",dt_obj)
print("type of dt:",type(dt_obj))