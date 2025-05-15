import os
from airSpace import AirSpace

class AirspaceLoader:
    REGIONS = {
        'Catalunya': {
            'nav': 'Cat_nav.txt',
            'seg': 'Cat_seg.txt',
            'aer': 'Cat_aer.txt'
        },
        'España': {
            'nav': 'Spa_nav.txt',
            'seg': 'Spa_seg.txt',
            'aer': 'Spa_aer.txt'
        },
        'Europe': {
            'nav': 'Eur_nav.txt',
            'seg': 'Eur_seg.txt',
            'aer': 'Eur_aer.txt'
        }
    }
    
    @classmethod
    def load_region(cls, region_name, data_dir='data'):
        """Load airspace data for a specific region"""
        if region_name not in cls.REGIONS:
            raise ValueError(f"Unknown region: {region_name}")
        
        region_files = cls.REGIONS[region_name]
        airspace = AirSpace(region_name)
        
        try:
            nav_path = os.path.join(data_dir, region_files['nav'])
            seg_path = os.path.join(data_dir, region_files['seg'])
            aer_path = os.path.join(data_dir, region_files['aer'])
            
            airspace.load_from_files(nav_path, seg_path, aer_path)
            return airspace
        except Exception as e:
            raise RuntimeError(f"Failed to load {region_name} airspace: {str(e)}")
    
    @classmethod
    def get_available_regions(cls):
        return list(cls.REGIONS.keys())