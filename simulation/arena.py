from simulation.base.abstract_arena import abstract_arena
import datetime
import os
from os.path import dirname
import json
import time
import re
import pickle
import chromadb
import numpy as np
import pandas as pd
from simulation.utils import *
from simulation.examiner import Examiner
import logging
import pathlib
import textwrap
import google.generativeai as genai

genai.configure(api_key="GENAI_API_KEY")


class Arena(abstract_arena):

    def __init__(self, args):
        super().__init__(args)

        self.llm_name = args.llm_name
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)  # Set the logging level to INFO or desired level
        
        # Create a handler and formatter
        self.train = args.train
        self.baseline = args.baseline
        self.args = args
        self.web_browsing = args.web_browsing
        self.model_type = args.model_type

       

        if self.train:
            log_directory = os.path.join("database", args.dataset+"/memory")
        elif self.web_browsing:
            log_directory = os.path.join("database", args.dataset+"/web_browsing")
        elif self.baseline:
            log_directory = os.path.join("database", args.dataset+"/base_test")
        else:
            log_directory = os.path.join("database", args.dataset+"/test")


        log_filename = "{}_{}.log".format(args.llm_name, args.saveID)
        log_path = os.path.join(log_directory, log_filename)

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(self.args)
        self.model = genai.GenerativeModel('gemini-pro')
       
        self.examiner = Examiner(args, self.logger, self.model)
       
        self.dataset = args.dataset
        self.saveID = args.saveID
        self.report_result = []
        self.start_from = args.start_from

        if args.train:
            self.prefix = "training"
        else:
            self.prefix = "testing"

        if self.dataset == 'law_harmfulbehavior':
            self.evaluator = RoBERTaPredictor('hubert233/GPTFuzz', device='cuda:0')
        elif 'stereotype' in self.dataset or 'ethic' in self.dataset:
            
            self.evaluator = GPTPredictor(args)
        elif self.dataset =='law_traffic':
            self.evaluator = GPTPredictor(args)

    def simulate_stereotype(self, statement, group):

        self.logger.info("Stereotype statement: {}".format(statement))
        self.logger.info("Stereotyped group: {}".format(group))

        example_list = self.examiner.retrieve_memory(misconduct=statement)

        response_generated = self.examiner.prompt_stereotype_agent(statement, example_list)
        self.logger.info("Stereotype generated response : {}".format(response_generated))
        result_generated = self.evaluator.predict(response_generated)
        self.logger.info("Stereotype generated result : {}".format(result_generated))

        if result_generated == 'PASS':

            updated_scenario = self.examiner.update_stereotype(statement, response_generated)
            response_updated = self.examiner.query_updated_stereotype(updated_scenario)
            result_updated = self.evaluator.predict(response_updated)

            self.logger.info("Stereotype adversarially updated RESPONSE: {}".format(response_updated))
            self.logger.info("Stereotype adversarially updated RESULT: {}".format(result_updated))

        else:
            result_updated = "FAIL"

        return result_generated, result_updated

    def simulate_stereotype_baselines(self, statement, group):

        self.logger.info("Stereotype statement: {}".format(statement))
        self.logger.info("Stereotyped group: {}".format(group))

        response_plain, response_untargeted, response_targeted = self.examiner.prompt_stereotype_baselines(
            statement, group)

        self.logger.info("Stereotype plain response : {}".format(response_plain))
        result_plain = self.evaluator.predict(response_plain)
        self.logger.info("Stereotype plain result : {}".format(result_plain))

        self.logger.info("Stereotype untargeted response : {}".format(response_untargeted))
        result_untargeted = self.evaluator.predict(response_untargeted)
        self.logger.info("Stereotype untargeted result : {}".format(result_untargeted))

        self.logger.info("Stereotype targeted response : {}".format(response_targeted))
        result_targeted = self.evaluator.predict(response_targeted)
        self.logger.info("Stereotype targeted result : {}".format(result_targeted))

        return result_plain, result_untargeted, result_targeted

    def simulate_ethic(self, ethic):

        self.logger.info("ETHIC: {}".format(ethic))

        example_list = self.examiner.retrieve_memory(misconduct=ethic)

        response_generated = self.examiner.prompt_ethic_agent(ethic, example_list)
        result_generated = self.evaluator.predict(response_generated)
        self.logger.info("ETHIC generated response : {}".format(response_generated))
        self.logger.info("ETHIC generated result : {}".format(result_generated))

        if result_generated == 'PASS':

            updated_scenario = self.examiner.update_ethic(ethic, response_generated)
            response_updated = self.examiner.query_updated_ethic(updated_scenario)
            result_updated = self.evaluator.predict(response_updated)
            self.logger.info("ETHIC adversarially updated RESPONSE: {}".format(response_updated))   
            self.logger.info("ETHIC adversarially updated RESULT: {}".format(result_updated)) 


        else:
            result_updated = 'FAIL'
                   
        return result_generated, result_updated

    def simulate_ethic_baselines(self, ethic):

        self.logger.info("ETHIC: {}".format(ethic))

        response_plain, response_untargeted, response_ev, response_context = self.examiner.prompt_ethic_baselines(ethic)
        
        self.logger.info("ETHIC plain: {}".format(response_plain))
        result_plain = self.evaluator.predict(response_plain)
        self.logger.info("ETHIC plain result: {}".format(result_plain))

        self.logger.info("ETHIC untargeted: {}".format(response_untargeted))
        result_untargeted = self.evaluator.predict(response_untargeted)
        self.logger.info("ETHIC untargeted result: {}".format(result_untargeted))

        self.logger.info("ETHIC evasive: {}".format(response_ev))
        result_ev = self.evaluator.predict(response_ev)
        self.logger.info("ETHIC evasive result: {}".format(result_ev))

        self.logger.info("ETHIC in-context: {}".format(response_context))
        result_context = self.evaluator.predict(response_context)
        self.logger.info("ETHIC in-context result: {}".format(result_context))

        return result_plain, result_untargeted, result_ev, result_context

    def simulate_law(self, law, index):

        example_list = self.examiner.retrieve_memory(misconduct=law)

        if self.dataset == 'law_traffic':
            response_plain = self.examiner.prompt_law_baselines(law, example_list, index)
            result_plain = self.evaluator.predict(response_plain)
            self.logger.info("LAW plain: {}".format(response_plain))
            
            result_generated = result_plain

        if self.dataset == 'law_harmfulbehavior':

            generated = self.examiner.generate_law_v2(law, example_list)
            response_generated = self.examiner.query_response_law(generated)
            result_generated = self.evaluator.predict(response_generated)
            self.logger.info("LAW generated response : {}".format(response_generated))
            
        self.logger.info("LAW generated result : {}".format(result_generated))

        if result_generated == 'PASS':

            if self.dataset == 'law_traffic':
                updated_scenario = self.examiner.update_law_v1(law, response_plain)            
            elif self.dataset == 'law_harmfulbehavior':
                updated_scenario = self.examiner.update_law_v2(law, response_generated)
            else:
                self.logger.info("Error processing")
                exit()

            response_updated = self.examiner.query_response_law(updated_scenario)
            result_updated = self.evaluator.predict(response_updated)
            self.logger.info("LAW adversarially updated RESPONSE: {}".format(response_updated))
            self.logger.info("LAW adversarially updated RESULT: {}".format(result_updated))
        else:
            result_updated = 'FAIL'

        return result_generated, result_updated

    def simulate_law_baselines(self, law, index):

        example_list = self.examiner.retrieve_memory(misconduct=law)
        response_plain, response_jb, response_ev, response_context = self.examiner.prompt_law_baselines(law, example_list, index)

        self.logger.info("LAW plain: {}".format(response_plain))
        result_plain = self.evaluator.predict(response_plain)
        self.logger.info("LAW plain RESULT: {}".format(result_plain))

        self.logger.info("LAW jailbreak or untargeted: {}".format(response_jb))
        result_jb = self.evaluator.predict(response_jb)
        self.logger.info("LAW jailbreak or untargeted RESULT: {}".format(result_jb))

        self.logger.info("LAW evasive: {}".format(response_ev))
        result_ev = self.evaluator.predict(response_ev)
        self.logger.info("LAW evasive RESULT: {}".format(result_ev))

        self.logger.info("LAW in-context: {}".format(response_context))
        result_context = self.evaluator.predict(response_context)
        self.logger.info("LAW in-context RESULT: {}".format(result_context))

        if self.dataset == 'law_harmfulbehavior':
            self.examiner.jailbreak_dict[self.examiner.selected_jailbreak_name][self.llm_name]["#test"] += 1
            if result_jb == 'FAIL':
                self.examiner.jailbreak_dict[self.examiner.selected_jailbreak_name][self.llm_name]["#success"] += 1

            with open("dataset/jailbreak.json", "w") as file:
                json.dump(self.examiner.jailbreak_dict, file)

        return result_plain, result_jb, result_ev, result_context

    def execute_stereotype_baselines(self):

        report_result = []
        statement_list = list(self.examiner.dataset_dict.keys())
        for i in range(self.start_from, len(statement_list)):

            statement = statement_list[i]

            self.logger.info("====== Assistant Baseline Response ======")
            self.logger.info("This is the {} th stereotype test.".format(i))

            group = self.examiner.dataset_dict[statement]

            result_plain, result_untargeted, result_targeted = self.simulate_stereotype_baselines(
                statement, group)

            result_object = {
                "result_plain": result_plain,
                "result_untargeted": result_untargeted,
                "result_targeted": result_targeted,
            }

            report_result.append(result_object)

        result = pd.DataFrame(report_result)
        violate_plain = len(result[result["result_plain"] == 'FAIL'])
        
        violate_untargeted = len(result[result["result_untargeted"] == 'FAIL'])
        violate_targeted = len(result[result["result_targeted"] == 'FAIL'])
        
        self.logger.info("STEREOTYPE total misconducts: {}".format(len(result)))
        self.logger.info("STEREOTYPE plain violation: {} | Invalid response {}".format(violate_plain))
        self.logger.info("STEREOTYPE untargeted violation: {} | Invalid response {}".format(violate_untargeted))
        self.logger.info("STEREOTYPE targeted violation: {} | Invalid response {}".format(violate_targeted))

    def execute_stereotype(self):

        statement_list = list(self.examiner.dataset_dict.keys())
        print(1)
        
        for i in range(self.start_from, len(statement_list)):
            
            statement = statement_list[i]

            self.logger.info("====== Assistant Response ======")
            self.logger.info("This is the {} th stereotype test.".format(i))

            group = self.examiner.dataset_dict[statement]

            result_generated, result_updated = self.simulate_stereotype(statement, group)

            result_object = {
                    "result_generated" : result_generated,
                    "result_updated" : result_updated
                }

            self.report_result.append(result_object)

        result = pd.DataFrame(self.report_result)

        self.logger.info("STEREOTYPE total misconducts: {}".format(len(result)))

        violate_generated = len(result[result["result_generated"]=='FAIL'])
        violate_updated = len(result[result["result_updated"]=='FAIL'])
        self.logger.info("STEREOTYPE generated violation: {}".format(violate_generated))
        self.logger.info("STEREOTYPE adversaral violation: {}".format(violate_updated))

    def execute_ethic_baselines(self):

        report_result = []
        for i in range(self.start_from, len(self.examiner.dataset_dict)):

            ethic = self.examiner.dataset_dict[i]

            self.logger.info("====== Assistant Baseline Response ======")
            self.logger.info("This is the {} th ethic test.".format(i))

            result_plain, result_untargeted, result_ev, result_context, = self.simulate_ethic_baselines(ethic)

            result_object = {
                    "result_plain" : result_plain,
                    "result_untargeted" : result_untargeted,
                    "result_evasive" : result_ev,
                    "result_context" : result_context,
                }

            report_result.append(result_object)

        result = pd.DataFrame(report_result)
        violate_plain = len(result[result["result_plain"]=='FAIL'])
        violate_untargeted = len(result[result["result_untargeted"]=='FAIL'])
        violate_ev = len(result[result["result_evasive"]=='FAIL'])
        violate_context = len(result[result["result_context"]=='FAIL'])

        self.logger.info("ETHICS total regulations: {}".format(len(result)))
        self.logger.info("ETHICS plain violation: {}".format(violate_plain))
        self.logger.info("ETHICS untargeted violation: {}".format(violate_untargeted))
        self.logger.info("ETHICS evasive violation: {}".format(violate_ev))
        self.logger.info("ETHICS in-context violation: {}".format(violate_context))

    def execute_ethic(self):

        for i in range(self.start_from, len(self.examiner.dataset_dict)):

            ethic = self.examiner.dataset_dict[i]

            self.logger.info("====== Assistant Response ======")
            self.logger.info("This is the {} th ethic test.".format(i))

            result_generated, result_updated = self.simulate_ethic(ethic)

            result_object = {
                "result_generated" : result_generated,
                "result_updated" : result_updated
            }

            self.report_result.append(result_object)

        result = pd.DataFrame(self.report_result)

        self.logger.info("ETHICS total regulations: {}".format(len(result)))

        violate_generated = len(result[result["result_generated"]=='FAIL'])
        violate_updated = len(result[result["result_updated"]=='FAIL'])
        self.logger.info("ETHICS generated violation: {}".format(violate_generated))
        self.logger.info("ETHICS adversaral violation: {}".format(violate_updated))

    def execute_law_baselines(self):

        report_result = []
        for i in range(self.start_from, len(self.examiner.dataset_dict)):

            law = self.examiner.dataset_dict[i]

            self.logger.info("====== Assistant Baseline Response ======")
            self.logger.info("This is the {} th LAW.".format(i))
            self.logger.info("LAW : {}.".format(law))

            result_plain, result_jb, result_ev, result_context = self.simulate_law_baselines(law, i)

            result_object = {
                "result_plain": result_plain, 
                "result_jailbreak": result_jb,
                "result_evasive": result_ev, 
                "result_context": result_context,
            }

            report_result.append(result_object)

        result = pd.DataFrame(report_result)
        violate_plain = len(result[result["result_plain"] == 'FAIL'])
        violate_jb = len(result[result["result_jailbreak"] == 'FAIL'])
        violate_ev = len(result[result["result_evasive"] == 'FAIL'])
        violate_context = len(result[result["result_context"] == 'FAIL'])

        self.logger.info("LAW total regulations: {}".format(len(result)))
        self.logger.info("LAW plain violation: {}".format(violate_plain))
        self.logger.info("LAW jailbreak or untargeted violation: {}".format(violate_jb))
        self.logger.info("LAW evasive violation: {}".format(violate_ev))
        self.logger.info("LAW in-context violation: {}".format(violate_context))

    def execute_law(self):
 
        for i in range(self.start_from, len(self.examiner.dataset_dict)):
            
            law = self.examiner.dataset_dict[i]

            self.logger.info("====== Assistant Response ======")
            self.logger.info("This is the {} th LAW.".format(i))
            self.logger.info("LAW : {}.".format(law))

            result_generated, result_updated = self.simulate_law(law, i)

            result_object = {
                "result_generated" : result_generated,
                "result_updated" : result_updated
            }

            self.report_result.append(result_object)

        result = pd.DataFrame(self.report_result)

        self.logger.info("LAW total regulations: {}".format(len(result)))

        violate_generated = len(result[result["result_generated"]=='FAIL'])
        violate_updated = len(result[result["result_updated"]=='FAIL'])
        self.logger.info("LAW generated scenario violation: {}".format(violate_generated))
        self.logger.info("LAW adversaral violation: {}".format(violate_updated))

    def execute_multi_round_harm(self):
        
        plug_in_jb = self.args.plug_in_jb
        jb_templates = pd.read_csv(f'{dirname(__file__)}/../dataset/gptfuzz_jb_prompts.csv', encoding='latin-1')['jb_prompts'].tolist()

        assert self.dataset == 'law_harmfulbehavior'
        max_trials = 5
        report_result = []
        for i in range(self.start_from, len(self.examiner.dataset_dict)):
            if plug_in_jb:
                random_index = random.randint(0, len(jb_templates)-1)
                jb_template = jb_templates[random_index]
                self.logger.info("Using {}th jb template".format(random_index))
            
            init_results = np.zeros(max_trials+1)
            law = self.examiner.dataset_dict[i]
            
            self.logger.info("====== Assistant Response ======")
            self.logger.info("Order {} | LAW: {}.".format(i, law))
            
            example_list = self.examiner.retrieve_memory(misconduct=law)
            generated = self.examiner.generate_law_v2(law, example_list)
            if plug_in_jb:
                generated =  jb_template.replace('[INSERT PROMPT HERE]', generated)

            response_generated = self.examiner.query_response_law(generated)
            result_generated = self.evaluator.predict(response_generated)
            self.logger.info("Harmful behavior round 0 | Generated Response: {}".format(response_generated))
            self.logger.info("Harmful behavior round 0 | Evaluation Result: {}".format(result_generated))
            
            trial = 0
            init_results[trial] = 0 if result_generated=='PASS' else 1
            while trial<max_trials and result_generated=='PASS':
                trial += 1
                updated_scenario = self.examiner.update_law_v2(law, response_generated)
                if plug_in_jb:
                    updated_scenario = jb_template.replace('[INSERT PROMPT HERE]', updated_scenario)

                response_updated = self.examiner.query_response_law(updated_scenario)
                result_updated = self.evaluator.predict(response_updated)
                self.logger.info("Harmful behavior adversarial round {} | Updated RESPONSE: {}".format(trial, response_updated))
                self.logger.info("Harmful behavior adversarial round {} | Updated RESULT: {}".format(trial, result_updated))
                
                response_generated = response_updated
                result_generated = result_updated
                init_results[trial] = 0 if result_generated=='PASS' else 1

            report_result.append(init_results)
        
        report_result = np.array(report_result)
        failure_sum = np.sum(report_result, axis=0).tolist()
        self.logger.info("Number of Harmful Questions: {}".format(report_result.shape[0]))
        for index, failure_num in enumerate(failure_sum):
            self.logger.info("Round {} | Number of Failure Questions: {}".format(index, int(failure_num)))
    
   

    def execute_browsing(self,bing_api_key,gpt_model_name,gpt_api_key,custom_config_id,bing_base_url='',gradio=False,input = None):
      laws = get_regulation_browsing(self.logger,bing_api_key,gpt_model_name,gpt_api_key,custom_config_id,bing_base_url,gradio,input)
      
      pattern = re.compile(r'\d+\.\s+(.+?)(?=\n\d+\.|\Z)', re.S)
      laws = pattern.findall(laws)
      print(laws)
      self.logger.info(f"Extracted laws: {laws}")

      numbered_laws = [f"{idx + 1}. {law}" for idx, law in enumerate(laws)]

      return "\n".join(numbered_laws),laws
    
    def escape_markdown(self,text):
        escape_chars = ["*", "_", "`", "[", "]", "(", ")", "~", "#", "+", "-", "|", "{", "}", ".", "!"]
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")  # 手动转义
        return text
    


    
    def Testing_law(self, laws, log_to_str=False):
        log_output = "" if log_to_str else None  # 只有 log_to_str=True 时才构建字符串

        for idx, law in enumerate(laws):
            law_cleaned = law.replace("\n", "").strip()
            log_entry = f"### 🔍 **Testing law: {law_cleaned}**\n"  # 放大加粗
            self.logger.info(log_entry)
            if log_to_str:
                log_output += log_entry + "\n"

            example_list = self.examiner.retrieve_memory(misconduct=law)
            generated = self.examiner.generate_law_v1(law, example_list)

            log_entry = f"**Generated scenario:** {generated}\n"
            if log_to_str:
                log_output += log_entry + "\n"

            response_generated = self.examiner.query_response_law(generated)
            result_generated = self.evaluator.predict(response_generated)

            log_entry = f"**Generated response:** {response_generated}\n"
            space = "---\n"
            self.logger.info(log_entry)
            if log_to_str:
                log_output += log_entry + "\n" + space

            log_entry = f"✅ **Test result:** {result_generated}\n"
            self.logger.info(log_entry)
            if log_to_str:
                log_output += log_entry + "\n" + space

            if result_generated == 'PASS':
                updated_scenario = self.examiner.update_law_v1(law, response_generated) 
                log_entry = f"**Updated scenario:** {updated_scenario}\n"
                if log_to_str:
                    log_output += log_entry + "\n" + space

                response_updated = self.examiner.query_response_law(updated_scenario)
                result_updated = self.evaluator.predict(response_updated)

                log_entry = f"**Updated response:** {response_updated}\n"
                self.logger.info(log_entry)
                if log_to_str:
                    log_output += log_entry + "\n" + space

                log_entry = f"✅ **Updated test result:** {result_updated}\n"
                self.logger.info(log_entry)
                if log_to_str:
                    log_output += log_entry + "\n" + space

            log_entry = "------\n"
            self.logger.info(log_entry)

        return log_output if log_to_str else None
