# page_classes/__init__.py

from page_classes.base_page import BasePage
from page_classes.log_analysis_page import LogAnalysisPage, log_analysis_page
from page_classes.splunk_config_page import SplunkConfigPage, splunk_config_page
from page_classes.corrections_page import CorrectionsPage, corrections_page
from page_classes.named_entity_recognition_page import NamedEntityRecognitionPage, named_entity_recognition_page

__all__ = [
    'BasePage',
    'LogAnalysisPage',
    'log_analysis_page',
    'SplunkConfigPage',
    'splunk_config_page',
    'CorrectionsPage',
    'corrections_page',
    'NamedEntityRecognitionPage',
    'named_entity_recognition_page'
]