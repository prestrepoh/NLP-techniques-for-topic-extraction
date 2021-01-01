import pandas as pd

class DataProcessor():

    @classmethod
    def obtain_boolean_mask_from_dataset(cls,dataset):
        dataset_topics = DataProcessor.get_column_indexes_as_list(dataset["Topic"])
        boolean_mask = DataProcessor.generate_boolean_mask(dataset["Topic"],dataset_topics).astype(int)
        return boolean_mask
    
    @classmethod
    def generate_boolean_mask(cls,item_lists, unique_items):
        bool_dict = {}
        
        # Loop through all the tags
        for i, item in enumerate(unique_items):
            
            # Apply boolean mask
            bool_dict[item] = item_lists.apply(lambda x: item in x)
                
        # Return the results as a dataframe
        return pd.DataFrame(bool_dict)
    
    @classmethod
    def get_column_indexes_as_list(cls,column):
        return DataProcessor.convert_list_to_series(column).value_counts().index.tolist()
    
    @classmethod
    def get_column_indexes(cls,column):
        return DataProcessor.convert_list_to_series(column).value_counts()

    @classmethod
    def convert_list_to_series(cls,list):
        return pd.Series([x for _list in list for x in _list])
    
    @classmethod
    def get_underrepresented_topics(cls,dataset,threshold):
        dataset_topics = DataProcessor.get_column_indexes(dataset["Topic"])
        return dataset_topics[dataset_topics < threshold].index.tolist()
    
    @classmethod
    def remove_topics_from_dataset(cls,dataset,boolean_mask,topics):
        dataset = dataset.join(boolean_mask)

        for topic in topics:
            dataset = dataset[~DataProcessor.row_contains_only_this_topic(dataset,topic)]
        
        dataset = dataset.drop(columns=topics)
        #dataset["list"] = dataset.iloc[:,9:].values.tolist()

        remaining_topics = dataset.iloc[:,9:].columns

        return dataset, remaining_topics

    @classmethod
    def row_contains_only_this_topic(cls,dataset, topic):
        row_contains_topic = (dataset[topic] == 1)
        row_has_only_one_topic = ((dataset.iloc[:,9:].drop(columns=topic) == 0).all(axis = 1))
        return row_contains_topic & row_has_only_one_topic







