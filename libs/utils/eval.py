"""
Evaluation of single result file.

Yu Fang - March 2019
"""

import os, glob, sys
import xml.dom.minidom
# from functools import cmp_to_key
from itertools import groupby
from .data_structure import *
from os.path import join as osj


class eval:
    def __init__(self, res_path, gt_table_info, pred_table_info):
        # 当前测评的表格的path
        self.return_result = None
        self.reg = False
        self.str = True

        self.resultFile = res_path
        self.inPrefix = os.path.split(res_path)[-1].split(".")[0]
        self.gene_ret_lst(gt_table_info, pred_table_info)

    @property
    def result(self):
        return self.return_result

    def gene_ret_lst(self,gt_table_info, pred_table_info):
        ret_lst = []
        for iou in [0.6]:
            temp,[total_correct_relation,total_gt_relation,total_res_relation] = self.compute_retVal(iou,gt_table_info, pred_table_info)
            ret_lst.extend([total_correct_relation,total_gt_relation,total_res_relation])
        self.return_result = ret_lst

    def compute_retVal(self, iou,gt_table_info, pred_table_info):

        ret = self.evaluate_result_str(gt_table_info, pred_table_info, iou)
        return ret
    
    @staticmethod
    def evaluate_result_str(gt_table_info, pred_table_info, iou_value, table_iou_value=0.8):
        total_gt_relation, total_res_relation, total_correct_relation = 0, 0, 0
        gt_table = Table(gt_table_info)
        ress_table = Table(pred_table_info)
        # set up the cell mapping for matching tables
        cell_mapping = gt_table.find_cell_mapping(ress_table, iou_value)
        # set up the adj relations, convert the one for result table to a dictionary for faster searching
        gt_AR = gt_table.find_adj_relations()
        total_gt_relation += len(gt_AR)
        
        res_AR = ress_table.find_adj_relations()
        total_res_relation += len(res_AR)
        
        if False:   # for DEBUG 
            Table.printCellMapping(cell_mapping)
            Table.printAdjacencyRelationList(gt_AR, "GT")
            Table.printAdjacencyRelationList(res_AR, "run")
        
        # Now map GT adjacency relations to result
        lMappedAR = []
        for ar in gt_AR:
            try:
                resFromCell = cell_mapping[ar.fromText]
                resToCell   = cell_mapping[ar.toText]
                #make a mapped adjacency relation
                lMappedAR.append(AdjRelation(resFromCell, resToCell, ar.direction))
            except:
                # no mapping is possible
                pass
        
        # compare two list of adjacency relation
        correct_dect = 0
        for ar1 in res_AR:
            for ar2 in lMappedAR:
                if ar1.isEqual(ar2):
                    correct_dect += 1
                    break

        total_correct_relation += correct_dect

        
        retVal = ResultStructure(truePos=total_correct_relation, gtTotal=total_gt_relation, resTotal=total_res_relation)
        return retVal,[total_correct_relation,total_gt_relation,total_res_relation]

