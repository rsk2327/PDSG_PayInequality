import os
import re
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def read_data(root_dir, data_folder, csv_file):
    print("Reading data from " + csv_file + "\n")
    file_dir = os.path.join(root_dir, data_folder, csv_file)
    return pd.read_csv(file_dir)

def merge_dataframe(dst_dataframe, new_part):
    if dst_dataframe is None:
        dst_dataframe = new_part.copy()
        return dst_dataframe
    else:
        dst_dataframe = pd.concat([dst_dataframe, new_part])
        return dst_dataframe

def compute_corelation(columns, src_dataframe):
    print("Computing Correlation matrix")
    return np.corrcoef(src_dataframe[columns].to_numpy(), rowvar = False)





if __name__ == "__main__":
    root_dir = "\\".join(os.path.dirname(__file__).split('\\')[:-1])
    data_folder = "reduced_paynet_data"
    task_folder = "task1"
    print('Analysis Starts\n')

    data_dir = os.path.join(root_dir, data_folder)
    all_annual_df = None
    all_annual_plus_df = None
    all_data_df = None
    cash_df = None

    '''
    Changable parameters
    '''
    data_merged = True
    total_analyzed = True
    year_analyzed = False
    sector_analyzed = True
    sector_annual_analyzed = False

    '''
    Data import
    '''
    if data_merged is False:
        for csv_file in os.listdir(data_dir):
            df = read_data(root_dir, data_folder, csv_file)
            all_annual_df = merge_dataframe(all_annual_df, df[['CalendarYear','IndustryName','Total Annual Remuneration']].dropna())
            all_annual_plus_df = merge_dataframe(all_annual_plus_df, df[['CalendarYear','IndustryName','Total Remuneration Plus']].dropna())
            cash_df = merge_dataframe(cash_df, df[['CalendarYear','IndustryName','Total Cash']])

            # These column has too few data points, which is regarded as not representative
            df = df.drop(columns = ['Benefit Values', 'Fixed Annual Remuneration', 'Total Earnings', 'Long Term Incentive Values', 'Short Term Variable Payments', 'Target Incentive Payment (%)', 'IndustrySegmentName', 'IndustrySectorName'])

            data_df = df.dropna()
            all_data_df = merge_dataframe(all_data_df, data_df)

        all_data_df.to_csv("no_nan_remuneration.csv", index=False)
        all_annual_df.to_csv("no_nan_annual_remuneration.csv", index = False)
        all_annual_plus_df.to_csv("no_nan_annual_remuneration_plus.csv", index = False)
        cash_df.to_csv("no_nan_cash.csv", index=False)

    else:
        all_data_df = read_data(root_dir, task_folder, "no_nan_remuneration.csv")
        all_annual_df = read_data(root_dir, task_folder, "no_nan_annual_remuneration.csv")
        all_annual_plus_df = read_data(root_dir, task_folder, "no_nan_annual_remuneration_plus.csv")
        cash_df = read_data(root_dir, task_folder, "no_nan_cash.csv")

    
    '''
    Total analysis
    '''
    if not total_analyzed:
    # correlation analysis
        all_columns = ['Base Salary', 'Total Annual Remuneration', 'Total Cash', 'Total Direct Compensation', 'Total Remuneration Plus']
        corelation_matrix = compute_corelation(all_columns, all_data_df)
        print(corelation_matrix)
        np.save('cor_mat.npy', corelation_matrix)

        # integrety analysis
        for temp in ['Total Annual Remuneration', 'Total Remuneration Plus']:
            plt.title(temp + " for all integrated data")
            plt.ylabel("Percentage per range")
            plt.xlabel('Range in logarithmic dollar')
            logit_data = np.log10(all_data_df[temp].tolist())
            logit_mean = np.mean(logit_data)
            logit_std = np.std(logit_data)

            n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
            y = norm.pdf(bins, logit_mean, logit_std)
            plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

            plt.xlim(left = 0, right = 10)
            plt.grid(True)
            plt.legend()
            # plt.show()
            plt.savefig("all_data_" + temp)
            plt.clf()

        temp = 'Total Cash'
        plt.title(temp + " for all integrated data")
        plt.ylabel("Percentage per range")
        plt.xlabel('Range in logarithmic dollar')
        logit_data = np.log10(cash_df[temp].tolist())
        logit_mean = np.mean(logit_data)
        logit_std = np.std(logit_data)

        n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
        y = norm.pdf(bins, logit_mean, logit_std)
        plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

        plt.xlim(left = 0, right = 10)
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig("all_data_" + temp)
        plt.clf()

    if not year_analyzed:
        # year level analysis
        all_year = pd.unique(all_data_df['CalendarYear']).tolist()
        all_year = range(min(all_year), max(all_year) + 1)
        year_mean = defaultdict(list)

        for curr_year in all_year:
            temp = 'Total Annual Remuneration'
            year_data_df = all_annual_df[all_annual_df['CalendarYear'] == curr_year]
            if year_data_df.shape[0] > 0:
                plt.title(temp + " for yearly integrated data")
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                if len(year_data_df[temp].tolist()) > 0:
                    logit_data = np.log10(year_data_df[temp].tolist())
                    logit_mean = np.mean(logit_data)
                    year_mean[temp].append(np.mean(year_data_df[temp].tolist()))
                    logit_std = np.std(logit_data)

                    n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                    y = norm.pdf(bins, logit_mean, logit_std)
                    plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                    plt.xlim(left = 0, right = 10)
                    plt.grid(True)
                    plt.legend()
                    # plt.show()
                    if not os.path.isdir(str(curr_year)):
                        os.mkdir(str(curr_year))
                    plt.savefig(os.path.join(str(curr_year), "year_data_" + temp))
                    plt.clf()
            else:
                year_mean[temp].append(0)

            temp = 'Total Remuneration Plus'
            year_data_df = all_annual_plus_df[all_annual_plus_df['CalendarYear'] == curr_year]
            if year_data_df.shape[0] > 0:
                plt.title(temp + " for yearly integrated data")
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                if len(year_data_df[temp].tolist()) > 0:
                    logit_data = np.log10(year_data_df[temp].tolist())
                    logit_mean = np.mean(logit_data)
                    year_mean[temp].append(np.mean(year_data_df[temp].tolist()))
                    logit_std = np.std(logit_data)

                    n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                    y = norm.pdf(bins, logit_mean, logit_std)
                    plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                    plt.xlim(left = 0, right = 10)
                    plt.grid(True)
                    plt.legend()
                    # plt.show()
                    if not os.path.isdir(str(curr_year)):
                        os.mkdir(str(curr_year))
                    plt.savefig(os.path.join(str(curr_year), "year_data_" + temp))
                    plt.clf()
            else:
                year_mean[temp].append(0)

            temp = 'Total Cash'
            year_data_df = cash_df[cash_df['CalendarYear'] == curr_year]
            plt.title(temp + " for yearly integrated data")
            plt.ylabel("Percentage per range")
            plt.xlabel('Range in logarithmic dollar')
            if len(year_data_df[temp].tolist()) > 0:
                logit_data = np.log10(year_data_df[temp].tolist())
                logit_mean = np.mean(logit_data)
                year_mean[temp].append(np.mean(year_data_df[temp].tolist()))
                logit_std = np.std(logit_data)

                n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                y = norm.pdf(bins, logit_mean, logit_std)
                plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                plt.xlim(left = 0, right = 10)
                plt.grid(True)
                plt.legend()
                # plt.show()
                if not os.path.isdir(str(curr_year)):
                    os.mkdir(str(curr_year))
                plt.savefig(os.path.join(str(curr_year), "year_data_" + temp))
                plt.clf()


        # linear regression of time with three variables
        plt.plot('Total year trend')
        plt.plot(all_year, year_mean['Total Annual Remuneration'], label = "Total Annual Remuneration")
        plt.plot(all_year, year_mean['Total Remuneration Plus'], label = 'Total Remuneration Plus')
        plt.plot(all_year, year_mean['Total Cash'], label = 'Total Cash')
        plt.savefig('Year Trend')

        
    if not sector_analyzed:
        all_industry = pd.unique(all_data_df['IndustryName']).tolist()
        all_year = pd.unique(all_data_df['CalendarYear']).tolist()
        all_year = range(min(all_year), max(all_year) + 1)
        for curr_industry in all_industry:
            industry = curr_industry.split('(')[0]
            industry = re.sub(r'\s+', '', industry)
            industry = re.sub('/+', '', industry)
            if not os.path.isdir(industry):
                os.mkdir(industry)
            industry_data_df = all_data_df[all_data_df['IndustryName'] == curr_industry]
            industry_cash_df = cash_df[cash_df['IndustryName'] == curr_industry]

            all_columns = ['Base Salary', 'Total Annual Remuneration', 'Total Cash', 'Total Direct Compensation', 'Total Remuneration Plus']
            corelation_matrix = compute_corelation(all_columns, industry_data_df)
            np.save(os.path.join(industry, 'cor_mat.npy'), corelation_matrix)

            for temp in ['Total Annual Remuneration', 'Total Remuneration Plus']:
                plt.title(temp + " for all data in " + industry)
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                logit_data = np.log10(industry_data_df[temp].tolist())
                logit_mean = np.mean(logit_data)
                logit_std = np.std(logit_data)

                n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                y = norm.pdf(bins, logit_mean, logit_std)
                plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                plt.xlim(left = 0, right = 10)
                plt.grid(True)
                plt.legend()
                # plt.show()
                plt.savefig(os.path.join(industry, "all_data_" + temp))
                plt.clf()

            temp = 'Total Cash'
            plt.title(temp + " for all data in " + industry)
            plt.ylabel("Percentage per range")
            plt.xlabel('Range in logarithmic dollar')
            logit_data = np.log10(industry_cash_df[temp].tolist())
            logit_mean = np.mean(logit_data)
            logit_std = np.std(logit_data)

            n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
            y = norm.pdf(bins, logit_mean, logit_std)
            plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

            plt.xlim(left = 0, right = 10)
            plt.grid(True)
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(industry, "all_data_" + temp))
            plt.clf()

    if not sector_annual_analyzed:
        all_industry = pd.unique(all_data_df['IndustryName']).tolist()
        all_year = pd.unique(all_data_df['CalendarYear']).tolist()
        all_year = range(min(all_year), max(all_year) + 1)
        for curr_industry in all_industry:    
            year_mean = defaultdict(list) 

            for curr_year in all_year:
                year_data_df = all_data_df[all_data_df['CalendarYear'] == curr_year]
                year_cash_df = cash_df[cash_df['CalendarYear'] == curr_year]
                industry_year_data_df = year_data_df[year_data_df['IndustryName'] == curr_industry]
                industry_year_cash_df = year_cash_df[year_cash_df['IndustryName'] == curr_industry]
                
                industry = curr_industry.split('(')[0]
                industry = re.sub(r'\s+', '', industry)
                industry = re.sub('/+', '', industry)
                if not os.path.isdir(os.path.join(str(curr_year), industry)):
                    os.mkdir(os.path.join(str(curr_year), industry))
            
                temp = 'Total Annual Remuneration'
                plt.title(temp + " for all data in " + industry + " of year " + str(curr_year))
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                if industry_year_data_df[temp].shape[0] > 0:
                    logit_data = np.log10(industry_year_data_df[temp].tolist())
                    logit_mean = np.mean(logit_data)
                    year_mean[temp].append(np.mean(industry_year_data_df[temp].tolist()))
                    logit_std = np.std(logit_data)

                    n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                    y = norm.pdf(bins, logit_mean, logit_std)
                    plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                    plt.xlim(left = 0, right = 10)
                    plt.grid(True)
                    plt.legend()
                    # plt.show()
                    plt.savefig(os.path.join(str(curr_year), industry, "all_data_" + temp))
                else:
                    year_mean[temp].append(0)
                plt.clf()


                temp = 'Total Remuneration Plus'
                plt.title(temp + " for all data in " + industry + " of year " + str(curr_year))
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                if industry_year_data_df[temp].shape[0] > 0:
                    logit_data = np.log10(industry_year_data_df[temp].tolist())
                    logit_mean = np.mean(logit_data)
                    year_mean[temp].append(np.mean(industry_year_data_df[temp].tolist()))
                    logit_std = np.std(logit_data)

                    n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                    y = norm.pdf(bins, logit_mean, logit_std)
                    plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                    plt.xlim(left = 0, right = 10)
                    plt.grid(True)
                    plt.legend()
                    # plt.show()
                    plt.savefig(os.path.join(str(curr_year), industry, "all_data_" + temp))
                else:
                    year_mean[temp].append(0)
                plt.clf()


                temp = 'Total Cash'
                plt.title(temp + " for all data in " + industry + " of year " + str(curr_year))
                plt.ylabel("Percentage per range")
                plt.xlabel('Range in logarithmic dollar')
                if industry_year_cash_df[temp].shape[0] > 0:
                    logit_data = np.log10(industry_year_cash_df[temp].tolist())
                    logit_mean = np.mean(logit_data)
                    year_mean[temp].append(np.mean(industry_year_cash_df[temp].tolist()))
                    logit_std = np.std(logit_data)

                    n, bins, _ = plt.hist(logit_data, bins=1000, density=True)
                    y = norm.pdf(bins, logit_mean, logit_std)
                    plt.plot(bins, y, label = "Mean: " + str(logit_mean) + " , Std: " + str(logit_std))

                    plt.xlim(left = 0, right = 10)
                    plt.grid(True)
                    plt.legend()
                    # plt.show()
                    plt.savefig(os.path.join(str(curr_year), industry, "all_data_" + temp))
                else:
                    year_mean[temp].append(0)
                plt.clf()

            plt.title(curr_industry + ' year trend')
            plt.plot(all_year, year_mean['Total Annual Remuneration'], label = 'Total Annual Remuneration')
            plt.plot(all_year, year_mean['Total Remuneration Plus'], label = 'Total Remuneration Plus')
            plt.plot(all_year, year_mean['Total Cash'], label = 'Total Cash')
            plt.savefig(os.path.join(industry, 'Year Trend'))


    np.load("cor_mat.npy")

    print('Analysis Ends')

