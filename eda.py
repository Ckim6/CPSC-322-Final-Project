import os 
from tabulate import tabulate
import mysklearn.myutils as myutils
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation
import pandas as pd
import csv

# 1. Arizona Cardinals
fileARI = os.path.join("input_data_csv", "sportsref_download_ARI.csv")
header = myutils.correct_header(fileARI) # header will be the same for each team/file
tableARI = myutils.correct_table(fileARI)

# 2. Atlanta Falcons
fileATL = os.path.join("input_data_csv", "sportsref_download_ATL.csv")
tableATL = myutils.correct_table(fileATL)

# 3. Baltimore Ravens
fileBAL = os.path.join("input_data_csv", "sportsref_download_BAL.csv")
tableBAL = myutils.correct_table(fileBAL)

# 4. Buffalo Bills
fileBUF = os.path.join("input_data_csv", "sportsref_download_BUF.csv")
tableBUF = myutils.correct_table(fileBUF)

# 5. Carolina Panthers
fileCAR = os.path.join("input_data_csv", "sportsref_download_CAR.csv")
tableCAR = myutils.correct_table(fileCAR)

# 6. Chicago Bears
fileCHI = os.path.join("input_data_csv", "sportsref_download_CHI.csv")
tableCHI = myutils.correct_table(fileCHI)

# 7. Cincinatti Bengals
fileCIN = os.path.join("input_data_csv", "sportsref_download_CIN.csv")
tableCIN = myutils.correct_table(fileCIN)

# 8. Cleveland Browns
fileCLE = os.path.join("input_data_csv", "sportsref_download_CLE.csv")
tableCLE = myutils.correct_table(fileCLE)

# 9. Dallas Cowboys
fileDAL = os.path.join("input_data_csv", "sportsref_download_DAL.csv")
tableDAL = myutils.correct_table(fileDAL)

# 10. Denver Broncos
fileDEN = os.path.join("input_data_csv", "sportsref_download_DEN.csv")
tableDEN = myutils.correct_table(fileDEN)

# 11. Detroit Lions
fileDET = os.path.join("input_data_csv", "sportsref_download_DET.csv")
tableDET = myutils.correct_table(fileDET)

# 12. Green Bay Packers 
fileGB = os.path.join("input_data_csv", "sportsref_download_GB.csv")
tableGB = myutils.correct_table(fileGB)

# 13. Houston Texans
fileHOU = os.path.join("input_data_csv", "sportsref_download_HOU.csv")
tableHOU = myutils.correct_table(fileHOU)

# 14, Indianapolis Colts
fileIND = os.path.join("input_data_csv", "sportsref_download_IND.csv")
tableIND = myutils.correct_table(fileIND)

# 15. Jacksonville Jaguars
fileJAC = os.path.join("input_data_csv", "sportsref_download_JAC.csv")
tableJAC = myutils.correct_table(fileJAC)

# 16. Kansas City Chiefs
fileKC = os.path.join("input_data_csv", "sportsref_download_KC.csv")
tableKC = myutils.correct_table(fileKC)

# 17. LA Chargers
fileLAC = os.path.join("input_data_csv", "sportsref_download_LAC.csv")
tableLAC = myutils.correct_table(fileLAC)

# 18. LA Rams
fileLAR = os.path.join("input_data_csv", "sportsref_download_LAR.csv")
tableLAR = myutils.correct_table(fileLAR)

# 19. Las Vegas Raiders
fileLV = os.path.join("input_data_csv", "sportsref_download_LV.csv")
tableLV = myutils.correct_table(fileLV)

# 20. Miami Dolphins
fileMIA = os.path.join("input_data_csv", "sportsref_download_MIA.csv")
tableMIA = myutils.correct_table(fileMIA)

# 21. Minnesota Vikings
fileMIN = os.path.join("input_data_csv", "sportsref_download_MIN.csv")
tableMIN = myutils.correct_table(fileMIN)

# 22. New England Patriots
fileNE = os.path.join("input_data_csv", "sportsref_download_NE.csv")
tableNE = myutils.correct_table(fileNE)

# 23. New Orleans Saints
fileNO = os.path.join("input_data_csv", "sportsref_download_NO.csv")
tableNO = myutils.correct_table(fileNO)

# 24. New York Giants
fileNYG = os.path.join("input_data_csv", "sportsref_download_NYG.csv")
tableNYG = myutils.correct_table(fileNYG)

# 25. New York Jets
fileNYJ = os.path.join("input_data_csv", "sportsref_download_NYJ.csv")
tableNYJ = myutils.correct_table(fileNYJ)

# 26. Philadelphia Eagles
filePHI = os.path.join("input_data_csv", "sportsref_download_PHI.csv")
tablePHI = myutils.correct_table(filePHI)

# 27. Pittsburgh Steelers
filePIT = os.path.join("input_data_csv", "sportsref_download_PIT.csv")
tablePIT = myutils.correct_table(filePIT)

# 28. Seattle Seahawks
fileSEA = os.path.join("input_data_csv", "sportsref_download_SEA.csv")
tableSEA = myutils.correct_table(fileSEA)

# 29. San Francisco 49ers
fileSF = os.path.join("input_data_csv", "sportsref_download_SF.csv")
tableSF = myutils.correct_table(fileSF)

# 30. Tamba Bay Buccaneers
fileTB = os.path.join("input_data_csv", "sportsref_download_TB.csv")
tableTB = myutils.correct_table(fileTB)

# 31. Tennesee Titans
fileTEN = os.path.join("input_data_csv", "sportsref_download_TEN.csv")
tableTEN = myutils.correct_table(fileTEN)

# 32. Washington Commanders
fileWAS = os.path.join("input_data_csv", "sportsref_download_WAS.csv")
tableWAS = myutils.correct_table(fileWAS)

# Combine all team data into one list
finalTable = tableARI + tableATL + tableBAL + tableBUF + tableCAR + tableCHI + tableCIN + \
    tableCLE + tableDAL + tableDEN + tableDET + tableGB + tableHOU + tableIND + tableJAC + tableKC + \
        tableLAC + tableLAR + tableLV + tableMIA + tableNE + tableNO + tableNYG + tableNYJ + tablePHI + \
            tablePIT + tableSEA + tableSF + tableTB + tableTEN + tableWAS

df = pd.DataFrame(finalTable, columns=header)
df = df[df.Year != '2022']
df = df[df.Year > '1966']
df.replace('', 'NP', inplace=True)
df.drop('Year', inplace=True, axis=1)
df.drop('Lg', inplace=True, axis=1)
df.drop('Tm', inplace=True, axis=1)
df.drop('W', inplace=True, axis=1)
df.drop('L', inplace=True, axis=1)
df.drop('T', inplace=True, axis=1)
df.drop('Div. Finish', inplace=True, axis=1)
df.drop('Coaches', inplace=True, axis=1)
df.drop('AV', inplace=True, axis=1)
df.drop('Passer', inplace=True, axis=1)
df.drop('Rusher', inplace=True, axis=1)
df.drop('Receiver', inplace=True, axis=1)
df.drop('out of', inplace=True, axis=1)
df.drop('PF', inplace=True, axis=1)
df.drop('PA', inplace=True, axis=1)
df.drop('PD', inplace=True, axis=1)

df.to_csv('input_data_cleaned.csv', index=False)
df = pd.read_csv('input_data_cleaned.csv')
print(df)