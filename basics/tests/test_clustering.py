
from basics.bubble_objects import Bubble3D
from basics.clustering import join_bubbles, threeD_overlaps


def test_join_single_overlap():
    '''
    Tests the joining function when there are multiple regions within a
    channel
    '''

    bub1 = Bubble3D.load_bubble("data/single_overlap_join_bubble_1.pkl")
    bub2 = Bubble3D.load_bubble("data/single_overlap_join_bubble_2.pkl")

    join_regions = join_bubbles([[bub1, bub2]])

    # There should be one less region in the output
    initial_num_regions = len(bub1.twoD_regions) + len(bub2.twoD_regions)
    final_num_regions = len(join_regions[0])

    assert final_num_regions == initial_num_regions - 1


def test_join_no_overlap():

    bub1 = Bubble3D.load_bubble("data/no_overlap_join_bubble_1.pkl")
    bub2 = Bubble3D.load_bubble("data/no_overlap_join_bubble_2.pkl")

    join_regions = join_bubbles([[bub1, bub2]])

    # There should be one less region in the output
    initial_num_regions = len(bub1.twoD_regions) + len(bub2.twoD_regions)
    final_num_regions = len(join_regions[0])

    assert final_num_regions == initial_num_regions


def test_join_single_overlap_3D():
    '''
    Testing the 3D overlap function with a pair that should be joined.
    '''

    bub1 = Bubble3D.load_bubble("data/single_overlap_join_bubble_1.pkl")
    bub2 = Bubble3D.load_bubble("data/single_overlap_join_bubble_2.pkl")

    bubbles, join_regions = \
        threeD_overlaps([[bub1, bub2]], overlap_frac=0.6, overlap_corr=0.7,
                        min_chan_overlap=2,
                        multiprocess=False, join_overlap_frac=0.6,
                        join_overlap_corr=0.6, min_multi_size=100,
                        n_jobs=None)

    # Should have the same output as in test_join_single_overlap

    # There should be one less region in the output
    initial_num_regions = len(bub1.twoD_regions) + len(bub2.twoD_regions)
    final_num_regions = len(join_regions[0])

    assert final_num_regions == initial_num_regions - 1
    assert len(bubbles) == 0


def test_join_no_overlap_3D():
    '''
    Testing the 3D overlap function with a pair that should be joined.
    '''

    bub1 = Bubble3D.load_bubble("data/no_overlap_join_bubble_1.pkl")
    bub2 = Bubble3D.load_bubble("data/no_overlap_join_bubble_2.pkl")

    bubbles, join_regions = \
        threeD_overlaps([[bub1, bub2]], overlap_frac=0.6, overlap_corr=0.7,
                        min_chan_overlap=2,
                        multiprocess=False, join_overlap_frac=0.6,
                        join_overlap_corr=0.6, min_multi_size=100,
                        n_jobs=None)

    # Should have the same output as in test_join_no_overlap

    # There should be one less region in the output
    initial_num_regions = len(bub1.twoD_regions) + len(bub2.twoD_regions)
    final_num_regions = len(join_regions[0])

    assert final_num_regions == initial_num_regions
    assert len(bubbles) == 0


def test_no_overlap_3D():
    '''
    Testing the 3D overlap function with a pair that should be joined.
    '''

    bub1 = Bubble3D.load_bubble("data/no_overlap_join_bubble_1.pkl")
    bub2 = Bubble3D.load_bubble("data/single_overlap_join_bubble_2.pkl")

    bubbles, join_regions = \
        threeD_overlaps([[bub1, bub2]], overlap_frac=0.6, overlap_corr=0.7,
                        min_chan_overlap=2,
                        multiprocess=False, join_overlap_frac=0.6,
                        join_overlap_corr=0.6, min_multi_size=100,
                        n_jobs=None)

    # No joining
    assert len(join_regions) == 0
    assert len(bubbles) == 2
