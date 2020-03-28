Predictive Flight Analysis
==========================

## About the project
The project was carried out as an internship project in AIR team. The project aims to deliver the solution
for reducing the count of flight cancellations and flight delays, to support both airlines and passengers.
The project was carried out in three phases.
- Preprocessing and Data cleanup
- Prediction phase
- Recommendation phase

## 1.0 Installation of software
For the successful installation of the software you need to follow the mentioned steps.
- If you wish to test the software using our data, go to section 1.1. Else, follow the steps below.
- Download the data from [US-BTS](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236) and 
[Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml?network=FR__ASOS) and save it to 
[UnProcessedData](UnProcessedData) folder.
- Clean and make the data ready for predictions using the steps of preprocessing as described in Section 2.0.
- Save the cleaned data to [Data](DashBoards/AllDashBoardsMerged/Data) folder.
- Go to [AllDashBoardsMerged](DashBoards/AllDashBoardsMerged) folder.
- Make sure that you have installed all libraries as mentioned in [libraries](libraries.md) document.
- Go to command prompt and change the directory to AllDashBoardsMerged folder.
- Run the command python index.py, and wait till the software starts running.
- If your pre-processed data is correct, you can view the dashboards on [127.0.0.1:8050](120.0.0.1:8050).

IMPORTANT NOTE 
- Do not temper with any other folder setup in [AllDashBoardsMerged](DashBoards/AllDashBoardsMerged) folder.

## 1.1 Software testing with preprocessed data
- You can download the cleaned data (tested working) from [OneDrive](https://amadeusworkplace.sharepoint.com/sites/PredictiveFlightAnalysis/Shared%20Documents/Forms/AllItems.aspx).
- Just copy the downloaded csv files to [Data](DashBoards/AllDashBoardsMerged/Data) folder.
- Make sure that you have installed all libraries as mentioned in [libraries](libraries.md) document.
- Go to command prompt and change directory to [Delay Models](Prediction-Phase-ML/DelayModels).
- Download the csv files from [One Drive](https://amadeusworkplace.sharepoint.com/:f:/r/sites/PredictiveFlightAnalysis/Shared%20Documents/Data%20For%20Delay%20Model%20To%20DB?csf=1&e=tY9bso)
and copy all that csv files in [Delay Models](Prediction-Phase-ML/DelayModels).
- Now, go back to command prompt and run the command "python saveDelayModelToDB1.py"
- After execution of this command run the command "python saveDelayModelToDB2.py"
- Go to command prompt and change the directory to AllDashBoardsMerged folder.
- Run the command python index.py, and wait till the software starts running.
- You can view the dashboards on [127.0.0.1:8050](120.0.0.1:8050).
<br>

## 2.0 Preprocessing
The preprocessing data for flights is downloaded 
from [US-BTS](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236) and 
for weather it is downloaded from [Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml?network=FR__ASOS).
Both websites provide the data freely. 

To use the data for predictions we needed to perform several cleaning operations on the data. The code-files for
preprocessing for cancellations and delays are in [PreProcessing-Cancellation](PreProcessing-Cancellation) and
[PreProcessing-Delay](Preprocessing-Delay) directories respectively.

The steps for preprocessing are mostly same for cancellations and delays. Broadly speaking, each step in 
both of the processing folder performs the following task:
- [Step 1](Preprocessing-Delay/Step-1_Traffic_Calculator.py): It calculates the traffic at origin airport at each day 
and at each hour and appends the calculated values in traffic column. 
- [Step 2](Preprocessing-Delay/Step-2_Weather_Cleanup.py): It takes input from downloaded weather excel file. It's
job is to select only that features, which are required for prediction. It also replaces the nan values with
hourly average weather entities. It also converts the coded values to their original numeric format.
- [Step 3](Preprocessing-Delay/Step-3_Optimizing_Weather.py): It take input from processed file from Step-2. This is an 
important step for successful implementation of Step 5. It merges the data and takes first value of each 
weather entity from each hour. This helps in merging the flights data and weather data.
- [Step 4](Preprocessing-Delay/Step-4_Flights_data_to_UTC.py): This step is carried out to convert the data to UTC
format. The flight data is in local time zone format and the weather data is UTC format. So, to properly
merge the flights and weather data, it is important fro both of them in same time format.
- [Step 5](Preprocessing-Delay/Step-5_Flight_Weather_merger.py): This is the final step of preprocessing. This takes
input from Step 3 and Step 4. It merges the flight with weather based on location, date and time. This data will be 
used for predictions now.

## 3.0 Exploratory Data Analysis
Exploratory data analysis was an important step to know what features should we use in prediction phase.
We made dashboards for the deciding whether the feature should be selected or not. These dashboards can be
located in [DelayAnalysisDashboards](DashBoards/DelayAnalysisDashboards) and 
[CancellationAnalysisDashboard](DashBoards/CancellationAnalysisDashboard). The description of each dashboard
is given below, while the working of code is explained using comments in the code.

[Delay Analysis Dashboards](DashBoards/DelayAnalysisDashboards)

- [OTP-Dashboard: Airport-wise](DashBoards/DelayAnalysisDashboards/app1.py): This dashboards shows you the
on-time performance values of the selected delay class. Example, if you select UA carrier and Delay type as 
15, it will show you the how many percentage of UA flights are having delay of less than 15 minutes. It means
we can take origin airport and destination airport as feature in delay prediction.

- [OTP-Dashboard: Daytime Wise](DashBoards/DelayAnalysisDashboards/app2.py): This dashboard will show you two heatmaps.
For the first heatmap, it is showing you hourwise On-Time performance values for selected airline on selected origin.
More red the color is, more good the on-time performance value is. The second heatmap will show you the mean departure 
delay of the selected carrier on selected origin. More red is the color, higher is the mean departure delay.
You can crosscheck the heatmaps. Where there is more red color in first heamap, there will be more blue color in
second heatmap. This means we can take the days and hour as the feature for delay prediction.

- [Mean Taxi-Out time by airport](DashBoards/DelayAnalysisDashboards/app3.py): This dashboard shows you the comparison of
mean taxi-out time of selected airline vs mean taxi-out time of all airline. If the mean taxi out time of selected airline
is greater than the mean taxi out time of all the airlines, it means that the flight is having bad taxi-out scheduling on that
particular airport. So we can consider taking Elapsed Time as a feature(elapsed time consists of taxi-out time
taxi-in time and air time). (Refer to definition section for understanding about taxi-out and taxi-in).

- [Taxi out time distributions](DashBoards/DelayAnalysisDashboards/app4.py): The dashboard shows us percentage distribution curve for 
taxi out time at origin airport and taxi in time at destination airport. The curve after selecting airline, origin and destination
shows us that this much percentage of flights go/arrive at this much taxi out/taxi in time. For example, a value of y axis as 0.3 and value of
x axis as 20 means, 30% of flights leave at 20 mins of taxi-out time. If there is a steep maxima, it means scheduking is good,
if the maxima is not steep, it means scheduling is poor.

- [Daytime Wise Taxi-Out dashboard](DashBoards/DelayAnalysisDashboards/app5.py): This dashboards shows us a heatmap of each day of week
and each hour of the day. More red the color is, higher is the taxi-out time of the selected carrier at selected airport. 

- [Block Time Dashboard](DashBoards/DelayAnalysisDashboards/app7.py): This dash board displays block time(refer definition section) for the selected flight. It shows the percentage distribution of 
block time vs the scheduled block time. There is an arrow head for representing the scheduled taxi-out time. If the maxima of actual taxi-out time percentage distribution is occuring after the arrowhead,
it means that flight is getting delayed because of poor block-time scheduling. Maybe, the airline should reschedule the flight timing to improve the block time and hence, getting less disruption.

- [Destination wise Arrival Delay Map](DashBoards/DelayAnalysisDashboards/app8.py): This dashboard show us a USA map, with circles around the 
destinations of selected airline. More big the circle is, higher are the number of delays.

 
[Cancellation Analysis Dashboards](DashBoards/CancellationAnalysisDashboard)

- [Cancellation Dashboard with reason distribution](DashBoards/CancellationAnalysisDashboard/edacan1.py): This dashboard displays the total cancellations
of selected flight number in selected date range. It also displays the cancellation reason distribution, if the flight is cancelled.

- [Cancellation Dashboard - Carrier Wise](DashBoards/CancellationAnalysisDashboard/edacan2.py): This dashboard is slightly different from the
dashboard above. It does not take route into account. It only takes carrier and flight number into account, so that you can see the cancellation of each
flight of the selected carrier.

- [Cancellation Dashboard - Route Wise cancellations](DashBoards/CancellationAnalysisDashboard/edacan3.py): Unlike the above dashboard, this dashboard 
only takes the origin destination pair into account. This enables the user to see which route has more cancellations, along with their distribution.

- [Cancellation Dashboard - Airline Route Wise](DashBoards/CancellationAnalysisDashboard/edacan4.py): This dashboard does not takes flight number into account.
It shows you the overall cancellation of the selected carrier on selected route and date range.

- [Traffic vs Cancellations Dashboard](DashBoards/CancellationAnalysisDashboard/edacan5.py): This dashboard shows the traffic vs the number of cancellations. For example traffic value is 6 and total cancellations are 
3, than it signifies that when the selected flight was about to depart, the traffic at that time at the airport was 6, and out of all the times when traffic was 6 and the flight was about to depart, 3 times this flight got cancelled.


## 4.0 Prediction and Recommendation
The prediction and recommendation engine requires the preprocessed data to execute successfully. For prediction
we used RIDGE regression in calculating delays and Random Forest in calculating cancellation probability. In both the 
cases we used the features calculated from exploratory data analysis. Also we used only that features/values
that are scheduled. We have not taken any actual(occuring in real time) value, so the model is plays well if you feed
only scheduled features

For the recommendation engine we controlled each feature that was causing the disruption, and saw the change
in the disruption. This gives the customer/user an idea that what feature should be controlled in order to nullify
the disruption. 

In case you want to use your own model, there is space provided in each code for the model training and testing.
To save the model to mongoDB you can refer to [SaveModelToDB](SaveModelToDB.py) file. Remember that if your model is having size greater than 16MB,
this code won't work, because mongoDB only takes documents that are less than 16MB in size. In that case you have to save model locally, the code for 
the same is provided in the prediction code files.
 
The code for prediction and recommendation can be located in [CancellationDashboards](DashBoards/CancellationDashboards)
and [DelayDashboards](DashBoards/DelayDashboards). The description of each dashboard is given below, while
the working of each code is explained using comments in the code.

[Delay Dashboards](DashBoards/DelayDashboards)

[Delay Predictions](DashBoards/DelayDashboards/DelayDashboard.py): This dashboard shows us the delay predictions(in minutes) for each day on which flight is scheduled.
For the flights getting delayed, it shows the delay reason distribution for each delay class. For example Weather_delay:66% means that, there are 66% chances that this flight
has got delayed due to weather.

[Delay Predictions with feature control](DashBoards/DelayDashboards/DelayReasonControlDashBoard2.py): This dashboard shows you the delay predictions(in minutes) for each day 
on which flight is scheduled. The additional feature is that now you can select a delay reason. It will assume that you have eliminated that feature somehow, and calculates the delay after
removing that feature. On the top of that it shows scheduled and estimated arrival time of flight and it also shows the decrement in delay after controlling each feature.

[Delay Predictions on Origin and Destination](DashBoards/DelayDashboards/DelayReasonControlDashBoardODwise.py): This dashboard takes the airline and route into account. It than shows a 
bar graph indicating the total delayed and total on time flights. It also has a feature of controlling the delay reason. You can clearly see the difference in bar graph after controlling any reason.
It also provides a dropdown list of delayed flights. If you select any flight, it will also show you the reason distribution of that delayed flight.


[Cancellation Dashboards](DashBoards/CancellationDashboards)

[Cancellation Predictions](DashBoards/CancellationDashboards/CancellationDashboard.py): This dashboard will show you the probability of cancellation of selected flight on each scheduled date of selected flight.
Also if the cancellation probability is greater than 0.5, it will show an arrowhead with the date and scheduled departure time of flight. And for the same case, its also shows the distribution of reason of cancellation.
For example NAS Cancellation Probaility :60% means that there are 60% chances that this flight will get cancelled due to National Air Security.

[Cancellation Predictions With Feature Control](DashBoards/CancellationDashboards/CancellationReasonControlDashboard.py): This dashboard will show you the probability of cancellation of selected flight on each scheduled date
of selected flight. The additional feature is that now you can select a delay reason. It will assume that you have eliminated that feature somehow, and calculates the cancellation 
probability after removing that feature. It also shows the decrement in the cancellation percentage after controlling that feature.

## 5.0 Definitions
Below are some definitions for proper understanding of codes and data.
- Cancellation and delay reason codes : There will be delay and cancellation codes present in data. Here's what they actually mean:

    1. Air Carrier: The cause of the cancellation or delay was due to circumstances within the airline's control (e.g. maintenance or crew problems, aircraft cleaning, baggage loading, fueling, etc.).

    2. Extreme Weather: Significant meteorological conditions (actual or forecasted) that, in the judgment of the carrier, delays or prevents the operation of a flight such as tornado, blizzard or hurricane.

    3. National Aviation System (NAS): Delays and cancellations attributable to the national aviation system that refer to a broad set of conditions, such as non-extreme weather conditions, airport operations, heavy traffic volume, and air traffic control.

    4. Late-arriving aircraft: A previous flight with same aircraft arrived late, causing the present flight to depart late.

    5. Security: Delays or cancellations caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.

- Traffic : The traffic is calculated from the data available to us, as it was not possible to check the data of flights from each airport in the world to the required airport. So we calculated the traffic 
from data available to us and assumed that the actual traffic would be in proportion to the calculated one. The traffic is calculated as the number of flights arriving to/departing from the selected airport
on the schedule departure time of that flight.

- Taxi-out time: Taxi-out time is the time between the flight leaving the gates and taking off.

- Taxi-in time: Taxi-in time is the time between landing of flight and arrival of flight at gates.

- Block time: The block time is defined as the total amount of time a flight takes from pushing back from departure gate to arriving at destination gate.

- Percentage distribution curve: It shows the percentage of y axis value occuring at corresponding x axis value. For example y:0.3 and x:2 means that the "2" value is occuring 30% of time in overall data.

- Heatmap:  A heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors. It is usually used for 
analyzing three variables at a time.