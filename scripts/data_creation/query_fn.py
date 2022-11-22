from influxdb import InfluxDBClient

def query_it(str1,client = InfluxDBClient(host="127.0.0.1", port=8086, database="lathrop")):
    x=client.query(str1)
    data = pd.DataFrame(x.get_points())
#     data.time = pd.to_datetime(data.time)
    return data

def query_betn(s_datetime,e_datetime,table,client = InfluxDBClient(host="127.0.0.1", port=8086, database="lathrop")):
    x = client.query(
        f"select * from {table} where time>='{s_datetime}' and time<'{e_datetime}'"
    )
    data=pd.DataFrame(x.get_points())
#     data.time=pd.to_datetime(data.time)
    return data




# a=query_it("select first(vibration) from raw")
# b=query_it("select last(vibration) from raw")
# print(a,"\n",b)


# vib_data_pd=query_betn("2022-09-29T15:00:00.000Z","2022-09-29T15:05:00.000Z","raw")
# vib_data_pd