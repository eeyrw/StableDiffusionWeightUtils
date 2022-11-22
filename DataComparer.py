import xml.etree.ElementTree as ET
import os
import sys
from terminaltables import AsciiTable
from functools import reduce
import textwrap
import traceback


class DataComparer(object):
    def __init__(self,dataRecord):
        #dataRecord=[{'name':[record name],
        # 'source:[record source path],
        # 'sourceFriendlyName':[Record source simple name],
        # 'dataDict':[Record field versus value dict]},...]
        self.compareFieldList=[]
        self.dataRecord=dataRecord
        self.checkResultListDict={}
        self.keepFullCompareResult=False
        return

    def customCompareFieldListList(self,compareFieldList):
        self.compareFieldList=compareFieldList

    def setKeepFullCompareResultFlag(self,flag):
        self.keepFullCompareResult=flag            

    def getShortExceptionInfo(self):
        exc_type, exc_value, exc_tb = sys.exc_info()
        tbe = traceback.TracebackException(exc_type, exc_value, exc_tb)
        return list(tbe.format_exception_only())[-1]
         
    def diffRecord(self,recordName,recordList):
        affectedFields=set()
        allFieldsList=list(reduce(lambda x,y :x|y,[set(item['dataDict'].keys()) for item in recordList]))
        currentCompareFieldList=[]
        tableData=[]
        tableData.append(['FieldName','Source', 'Value'])

        if self.compareFieldList==[]:
            currentCompareFieldList=allFieldsList
        else:
            currentCompareFieldList=list(set(allFieldsList)&set(self.compareFieldList))


        diffCount=0
        for compareField in currentCompareFieldList:
            valueItemDict={}
            for record in recordList:
                recordDataDict=record['dataDict']
                if compareField in recordDataDict.keys():
                    fieldValue=recordDataDict[compareField]
                    valueItemDict.setdefault(fieldValue,[]).append(record)
                else:
                    valueItemDict.setdefault('MISSING',[]).append(record)

            if len(valueItemDict.items())>1:
                diffCount+=1
                for k,v in valueItemDict.items():
                    affectedFields.add(compareField)
                    if self.keepFullCompareResult:
                        valueText=k
                    else:
                        valueText=textwrap.shorten(k,width=35)
                    tableData.append([compareField,reduce((lambda x, y: x+'\n'+y),[item['sourceFriendlyName'] for item in v]), valueText])              

        table = AsciiTable(tableData)  
        table.inner_row_border = True  
        table.title = recordName
        return diffCount,affectedFields,table.table,tableData      

    def compare(self):
        diffResultTextList=[]
        diffResultTableDataList=[]
        variableDataRecordListDict={}
        for record in self.dataRecord:
            variableDataRecordListDict.setdefault(record['name'],[]).append(record)
        
        for k,v in variableDataRecordListDict.items():
            diffCount,affectedFields,diffTextResult,tableData=self.diffRecord(k,v)
            if diffCount>0:
                diffResultTextList.append(diffTextResult)
                diffResultTableDataList.append((k,affectedFields,tableData))
        return diffResultTextList,diffResultTableDataList
