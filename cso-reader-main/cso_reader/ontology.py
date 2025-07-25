import pickle
import os
import csv as co
import urllib.request
import json
from collections import deque
import difflib

from config import Config
from misc import print_header, download_file


class Ontology:
    """ A simple abstraction layer for using the Computer Science Ontology """

    def __init__(self, load_ontology = True, silent = False):
        """ Initialising the ontology class
        """
        self.silent = silent
        self.topics = dict()
        self.topics_wu = dict()
        self.broaders = dict()
        self.narrowers = dict()
        self.same_as = dict()
        self.primary_labels = dict()
        self.primary_labels_wu = dict()
        self.topic_stems = dict()
        self.all_broaders = dict()
        self.level = dict()
        self.graph = None

        self.config = Config()

        self.ontology_attr = ('topics',
                              'topics_wu',
                              'broaders',
                              'narrowers',
                              'same_as',
                              'primary_labels',
                              'primary_labels_wu',
                              'topic_stems',
                              'all_broaders',
                              'level')

        if load_ontology:
            self.load_ontology_pickle()


# =============================================================================
#     CSO DICT
# =============================================================================


    def from_single_items_to_cso(self):
        """ Function that returns a single dictionary containing all relevant
        values for the ontology.
        """
        return {attr: getattr(self, attr) for attr in self.ontology_attr}

    def from_cso_to_single_items(self, cso):
        """ Function that fills all the single relevant variables in this class.
        """
        for attr in self.ontology_attr:
            try:
                setattr(self, attr, cso[attr])
            except KeyError:
                ValueError("Key {} not found in the ontology".format(attr))


    def load_ontology_pickle(self):
        """ Function that loads CSO.
        This file has been serialised using Pickle allowing to be loaded quickly.
        """
        self.check_ontology()
        ontology = pickle.load(open(self.config.get_cso_pickle_path(), "rb" ))
        self.from_cso_to_single_items(ontology)
        if not self.silent:
            print("Computer Science Ontology loaded.")


    def get_primary_label(self, topic):
        """ Function that returns the primary (preferred) label for a topic.
        If this topic belongs to a cluster.

        Args:
            topic (string): Topic to analyse.

        Returns:
            topic (string): primary label of the analysed topic.
        """

        try:
            topic = self.primary_labels[topic]
        except KeyError:
            pass

        return topic


    def get_primary_label_wu(self, topic):
        """ Function that returns the primary (preferred) label for a topic *with underscore*. If this
        topic belongs to a cluster.

        Args:
            topic (string): Topic to analyse.

        Returns:
            topic (string): primary label of the analysed topic with underscore.
        """

        try:
            topic = self.primary_labels_wu[topic]
        except KeyError:
            pass

        return topic


    def get_topic_wu(self, topic):
        """ Function that returns the topic label (without underscore) from its underscored version.

        Args:
            topic (string): Topic to analyse.

        Returns:
            topic (string): primary label of the analysed topic with underscore.
        """

        try:
            topic = self.topics_wu[topic]
        except KeyError:
            pass

        return topic


    def climb_ontology(self, found_topics, climb_ont):
        """ Function that climbs the ontology. This function might retrieve
            just the first broader topic or the whole branch up until root
        Args:
            found_topics (list): It contains the topics found with string similarity.
            climb_ont (string): either "first" or "all" for selecting "just the first broader topic"
            or climbing the "whole tree".
        Returns:
            found_topics (dictionary): containing the found topics with their similarity and the
            n-gram analysed.
        """

        all_broaders = dict()
        num_narrowers = 1

        if climb_ont == 'first':
            all_broaders = self.get_broader_of_topics(found_topics, all_broaders)
        elif climb_ont == 'all':
            while True:
                # """
                # recursively adding new broaders based on the current list of topics.
                #Broaders var increases each iteration. It stops when it does not change anymore.
                # """
                all_broaders_back = all_broaders.copy()
                all_broaders = self.get_broader_of_topics(found_topics, all_broaders)
                if all_broaders_back == all_broaders:  # no more broaders have been found
                    break
        elif climb_ont == 'no':
            return dict() #it is empty at this stage
        else:
            raise ValueError("Error: Field climb_ontology must be 'first', 'all' or 'no'")


        inferred_topics = dict()
        for broader, narrowers in all_broaders.items():
            if len(narrowers) >= num_narrowers:
                broader = self.get_primary_label(broader)
                if broader not in inferred_topics:
                    inferred_topics[broader] = {'matched': len(narrowers), 'broader of': list(narrowers)}
                else: # this branch folds when we have same as
                    this_broader_narrowers = set(inferred_topics[broader]['broader of'])
                    this_broader_narrowers = this_broader_narrowers.union(narrowers)
                    inferred_topics[broader] = {'matched': len(this_broader_narrowers), 'broader of': list(this_broader_narrowers)}

        return inferred_topics


    def get_broader_of_topics(self, found_topics, all_broaders=dict()):
        """ Function that returns all the broader topics for a given set of topics.
            It analyses the broader topics of both the topics initially found in the paper, and the broader topics
            found at the previous iteration.
            It incrementally provides a more comprehensive set of broader topics.

        Args:
            found_topics (list): It contains the topics found with string similarity.
            all_broaders (dictionary): It contains the broader topics found in the previous run. Otherwise an empty object.

        Returns:
            all_broaders (dictionary): contains all the broaders found so far, including the previous iterations.
        """
        topics = list(found_topics) + list(all_broaders.keys())
        for topic in topics:
            try:
                broaders = self.broaders[topic]
                for broader in broaders:
                    if broader not in all_broaders:
                        all_broaders[broader] = set()
                    all_broaders[broader].add(topic)
                    if topic in all_broaders:
                        all_broaders[broader] = all_broaders[broader].union(all_broaders[topic])

            except KeyError:
                pass

        return all_broaders


    def get_all_broaders_of_topic(self, topic):
        """ Function that returns all the broader topics up to the root.

        Args:
            topic (string): the input topic.

        Returns:
            all_broaders (list): contains all the broaders topics of topic within CSO.
        """
        all_broaders = list()
        try:
            all_broaders = self.all_broaders[topic]
        except KeyError:
            pass

        return all_broaders
    
    
    def get_all_descendants_of_topics(self, topics):
        """
        Finds all the descendants of a given list (or set) of topics. 

        Parameters
        ----------
        topics : list (or set)
            List of topics of which identifying the descendants.

        Raises
        ------
        ValueError
            Error is raised when a different type of datase.

        Returns
        -------
        list
            The unique list of all descendants of the input topics.

        """
        
        if type(topics) == str:
            return self.get_all_descendants_of_topic(topics)
        elif type(topics) == list or type(topics) == set:
            descendants = []
            for topic in topics:
                descendants.extend(self.get_all_descendants_of_topic(topic))
            return list(set(descendants))
        else:
            raise TypeError("Error: The type of 'topics' must be either list or set.")
    
    
    def get_all_descendants_of_topic(self, topic):
        """
        Identifies all the descendants of a given topic of CSO

        Parameters
        ----------
        topic : str
            the topic.

        Raises
        ------
        TypeError
            raises the error if the topic is not a string.

        Returns
        -------
        list
            the list of descendant topics of 'topic'.

        """
        
        
        if type(topic) != str: 
            raise TypeError("Error: The type of 'topic' must be str.")
        
        if topic not in self.topics:
            raise ValueError(f"Error: The topic '{topic}' is not available in this version of the Ontology.")
        
        set_of_descendants = set()
        queue = deque() 
        queue.append(topic)
        
        while len(queue) > 0:
            dequeued = queue.popleft()
            set_of_descendants.add(dequeued)
            if dequeued in self.narrowers:
                narrower_concepts = self.narrowers[dequeued]
                for narrower_concept in narrower_concepts:
                    queue.append(narrower_concept)

        return list(set_of_descendants)


    def find_closest_matches(self, word):
        """ Function that finds the closest match of a given topic (by looking at the topic stems)
        """
        list_of_topics = list()
        if word[:4] in self.topic_stems:
            list_of_topics =  self.topic_stems[word[:4]]

        return list_of_topics




# =============================================================================
#     CONFIG and SETUP
# =============================================================================


    def check_ontology(self):
        """ Function that checks if the ontology is available.
        If not, it will check if a csv version exists and then it will create the pickle file.
        """
        if not os.path.exists(self.config.get_cso_pickle_path()):
            print("Ontology pickle file is missing.")

            if not os.path.exists(self.config.get_cso_path()):
                print("The source file of the Computer Science Ontology is missing. Attempting to download it now...")
                self.__download_ontology()

            self.__load_cso_from_csv()



    def __load_cso_from_csv(self):
        """Function that loads the CSO from the file in a dictionary.
           In particular, it load all the relationships organised in boxes:
               - topics, the list of topics
               - broaders, the list of broader topics for a given topic
               - narrowers, the list of narrower topics for a given topic
               - same_as, all the siblings for a given topic
               - primary_labels, all the primary labels of topics, if they belong to clusters
               - topics_wu, topic with underscores
               - primary_labels_wu, primary labels with underscores
               - topic_stems, groups together topics that start with the same 4 letters
        """

        print("Extracting and converting ontology.")
        with open(self.config.get_cso_path(), 'r') as onto_file:

            ontology = co.reader(onto_file, delimiter=';')

            for triple in ontology:
                if triple[1] == 'klink:broaderGeneric':
                    # loading broader topics
                    if triple[2] in self.broaders:
                        self.broaders[triple[2]].append(triple[0])
                    else:
                        self.broaders[triple[2]] = [triple[0]]

                    # loading narrower topics
                    if triple[0] in self.narrowers:
                        self.narrowers[triple[0]].append(triple[2])
                    else:
                        self.narrowers[triple[0]] = [triple[2]]
                elif triple[1] == 'klink:relatedEquivalent':
                    if triple[2] in self.same_as:
                        self.same_as[triple[2]].append(triple[0])
                    else:
                        self.same_as[triple[2]] = [triple[0]]
                elif triple[1] == 'rdfs:label':
                    self.topics[triple[0]] = True
                    self.topics_wu[triple[0].replace(" ", "_")] = triple[0]
                elif triple[1] == 'klink:primaryLabel':
                    self.primary_labels[triple[0]] = triple[2]
                    self.primary_labels_wu[triple[0].replace(" ", "_")] = triple[2].replace(" ", "_")


            self.__generate_topic_stems()
            self.__get_all_branches()
            self.__get_levels()


            with open(self.config.get_cso_pickle_path(), 'wb') as cso_file:
                print("Creating ontology pickle file from a copy of the CSO Ontology found in",self.config.get_cso_path())
                pickle.dump(self.from_single_items_to_cso(), cso_file)



    def __generate_topic_stems(self):
        """ Function that generates all topics stems which will be useful in the syntactic module
        """
        for topic, _ in self.topics.items():
            if topic[:4] not in self.topic_stems:
                self.topic_stems[topic[:4]] = list()
            self.topic_stems[topic[:4]].append(topic)


    def __get_all_branches(self):
        """ Function that identifies all broaders of a given topic.
        """
        for topic, _ in self.topics.items():
            this_topic_broaders = list()
            queue = deque()
            queue.append(topic)
            while len(queue) > 0:
                dequeued = queue.popleft()
                if dequeued in self.broaders:
                    broaders = self.broaders[dequeued]
                    for broader in broaders:
                        queue.append(broader)
                        this_topic_broaders.append(broader)
            self.all_broaders[topic] = list(set(this_topic_broaders))

    def __get_levels(self):
        """ Function that computes the depth level of all topics in CSO
        """
        
        self.level = {topic:1 for topic,_ in self.topics.items()}
        for concept, value in self.level.items(): 
            queue = deque() 
            max_depth = value
            
            queue.append({"t":concept,"d":value})
            
            while len(queue) > 0:
                dequeued = queue.popleft()
                if dequeued["t"] in self.broaders:
                    broaders = self.broaders[dequeued["t"]]
                    new_depth = dequeued["d"]+1
                    if new_depth > max_depth:
                        max_depth = new_depth
                    for broader in broaders:
                        queue.append({"t":broader,"d":dequeued["d"]+1})

            self.level[concept] = max_depth



    def __download_ontology(self):
        """ Function that allows to download the latest version of the ontology.
            If older versions of the ontology (both csv and pickle) are available they will be deleted.
        """
        try:
            os.remove(self.config.get_cso_pickle_path())
        except FileNotFoundError:
            pass

        try:
            os.remove(self.config.get_cso_path())
        except FileNotFoundError:
            pass

        ontology_remote_url, last_version = self.retrieve_url_of_latest_version_available()
        print("Downloading the Computer Science Ontology from {}".format(ontology_remote_url))
        task_completed = download_file(ontology_remote_url, self.config.get_cso_path())
        self.config.set_cso_version(last_version)
        return task_completed


    def update(self, force = False):
        """ This funciton updates the ontology.

        Args:
            force (boolean): If false, it checks if a newer version is available.
                If false, it will delete all files and download the most recent version.
        """
        print_header("ONTOLOGY")
        if force:
            print("Updating the ontology file")
            self.__download_ontology()
            self.__load_cso_from_csv()

        else:
            last_version = self.retrieve_latest_version_available()
            if last_version > self.config.get_ontology_version():
                print("Updating the ontology file")
                self.__download_ontology()
                self.__load_cso_from_csv()
            else:
                print("The ontology is already up to date.")


    def setup(self):
        """ Function that sets up the ontology.
        """
        print_header("ONTOLOGY")

        if not os.path.exists(self.config.get_cso_pickle_path()):

            if not os.path.exists(self.config.get_cso_path()):

                task_completed = self.__download_ontology()

                if task_completed:
                    print("Ontology file downloaded successfully.")
                else:
                    print("We were unable to complete the download of the ontology.")

            self.__load_cso_from_csv()

            if os.path.exists(self.config.get_cso_pickle_path()):
                print("Ontology file created successfully.")

        else:
            print("Nothing to do. The ontology file is already available.")


    def retrieve_url_of_latest_version_available(self):
        """ Function that retireves the version number of the latest ontology.
        """
        version = self.retrieve_latest_version_available()
        composite_url = "{0}/version-{1}/cso_v{1}.csv".format(self.config.get_cso_remote_url(),version)
        with urllib.request.urlopen(self.config.get_cso_version_logger_url()) as url:
            data = json.loads(url.read().decode())
            if "last_version" in data and "url" in data['last_version']:
                composite_url = data['last_version']["url"]

        return composite_url, version


    def retrieve_latest_version_available(self):
        """ Function that retireves the version number of the latest ontology.
        """
        version = "0.0"
        with urllib.request.urlopen(self.config.get_cso_version_logger_url()) as url:
            data = json.loads(url.read().decode())
            if "last_version" in data and "version" in data['last_version']:
                version = data['last_version']["version"]

        return version
        
    def get_top_n_similar(self, term, n=5):
        """
        Returns the top n most similar ontology topics based on string similarity.
        """
        # Get the list of all topic labels
        topic_labels = list(self.topics.keys())

        # Use difflib's get_close_matches
        matches = difflib.get_close_matches(term.lower(), topic_labels, n=n, cutoff=0.0)

        # Optionally, score each match
        scored_matches = [(match, difflib.SequenceMatcher(None, term.lower(), match).ratio())
                          for match in matches]

        # Sort by similarity score
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        return scored_matches


    def version(self):
        """ Function that returns the current version of the ontology available in this classifier
            It also checks whether there is a more up-to-date version.
        """
        print_header("ONTOLOGY")
        print("CSO ontology version {}".format(self.config.get_ontology_version()))

        last_version = self.retrieve_latest_version_available()

        if last_version > self.config.get_ontology_version():
            print("A more recent version ({}) of the Computer Science Ontology is available.".format(last_version))
            print("You can update this package by running the following instructions:")
            print("1) import cso_classifier as cc")
            print("2) cc.update()")
        elif last_version == self.config.get_ontology_version():
            print("The version of the CSO Ontology you are using is already up to date.")
        elif last_version < self.config.get_ontology_version():
            print("Something is not right. The version you are using ({}) is ahead compared to the latest available ({}).".format(self.config.get_ontology_version(),last_version))

# Example usage
if __name__ == "__main__":
    try:
        ontology = Ontology()
        primary_label = ontology.get_primary_label("machine learning")  # Returns the primary label for "machine learning"
        print(f"Primary label for 'machine learning': {primary_label}")
        broader_topics = ontology.get_all_broaders_of_topic("machine learning")  # Returns a list of broader topics
        print(f"Broader topics for 'machine learning': {broader_topics}")
    except Exception as e:
        print(f"An error occurred: {e}")