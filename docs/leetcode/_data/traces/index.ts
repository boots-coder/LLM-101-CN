import type { AlgoTrace } from './types'

import permutations from './permutations'
import twoSum from './two-sum'
import threeSum from './three-sum'
import longestSubstring from './longest-substring-without-repeating'
import searchRotated from './search-rotated-array'
import subarraySumK from './subarray-sum-equals-k'
import houseRobber from './house-robber'
import uniquePaths from './unique-paths'
import dailyTemperatures from './daily-temperatures'
import kthLargest from './kth-largest-element'
import reverseLinkedList from './reverse-linked-list'
import inorderTraversal from './inorder-traversal'
import levelOrderTraversal from './level-order-traversal'
import implementTrie from './implement-trie'
import numberOfIslands from './number-of-islands'

// Agent 1: ArrayBoardViz
import findAnagrams from './find-anagrams'
import containerWithMostWater from './container-with-most-water'
import minimumWindowSubstring from './minimum-window-substring'
import findPeakElement from './find-peak-element'
import singleNumber from './single-number'

// Agent 2: stack/heap/array mix
import trappingRainWater from './trapping-rain-water'
import medianTwoSortedArrays from './median-two-sorted-arrays'
import productExceptSelf from './product-except-self'
import topKFrequent from './top-k-frequent'
import largestRectangleHistogram from './largest-rectangle-histogram'

// Agent 3: heap/DP
import findMedianDataStream from './find-median-data-stream'
import climbingStairs from './climbing-stairs'
import longestIncreasingSubsequence from './longest-increasing-subsequence'
import wordBreak from './word-break'
import editDistance from './edit-distance'

// Agent 4: DP/backtracking
import longestCommonSubsequence from './longest-common-subsequence'
import longestPalindromicSubstring from './longest-palindromic-substring'
import combinationSum from './combination-sum'
import nQueens from './n-queens'
import wordSearch from './word-search'

// Agent 5: graph/linked-list
import courseSchedule from './course-schedule'
import rottingOranges from './rotting-oranges'
import linkedListCycleII from './linked-list-cycle-ii'
import mergeKSortedLists from './merge-k-sorted-lists'
import lruCache from './lru-cache'

// Agent 6: tree/Trie
import binaryTreeMaxPathSum from './binary-tree-max-path-sum'
import lowestCommonAncestor from './lowest-common-ancestor'
import serializeDeserializeTree from './serialize-deserialize-tree'
import flattenTreeToList from './flatten-tree-to-list'
import rightSideView from './right-side-view'
import wordSearchII from './word-search-ii'

export const traces: Record<string, AlgoTrace[]> = {
  permutations,
  'two-sum': twoSum,
  'three-sum': threeSum,
  'longest-substring-without-repeating': longestSubstring,
  'search-rotated-array': searchRotated,
  'subarray-sum-equals-k': subarraySumK,
  'house-robber': houseRobber,
  'unique-paths': uniquePaths,
  'daily-temperatures': dailyTemperatures,
  'kth-largest-element': kthLargest,
  'reverse-linked-list': reverseLinkedList,
  'inorder-traversal': inorderTraversal,
  'level-order-traversal': levelOrderTraversal,
  'implement-trie': implementTrie,
  'number-of-islands': numberOfIslands,
  'find-anagrams': findAnagrams,
  'container-with-most-water': containerWithMostWater,
  'minimum-window-substring': minimumWindowSubstring,
  'find-peak-element': findPeakElement,
  'single-number': singleNumber,
  'trapping-rain-water': trappingRainWater,
  'median-two-sorted-arrays': medianTwoSortedArrays,
  'product-except-self': productExceptSelf,
  'top-k-frequent': topKFrequent,
  'largest-rectangle-histogram': largestRectangleHistogram,
  'find-median-data-stream': findMedianDataStream,
  'climbing-stairs': climbingStairs,
  'longest-increasing-subsequence': longestIncreasingSubsequence,
  'word-break': wordBreak,
  'edit-distance': editDistance,
  'longest-common-subsequence': longestCommonSubsequence,
  'longest-palindromic-substring': longestPalindromicSubstring,
  'combination-sum': combinationSum,
  'n-queens': nQueens,
  'word-search': wordSearch,
  'course-schedule': courseSchedule,
  'rotting-oranges': rottingOranges,
  'linked-list-cycle-ii': linkedListCycleII,
  'merge-k-sorted-lists': mergeKSortedLists,
  'lru-cache': lruCache,
  'binary-tree-max-path-sum': binaryTreeMaxPathSum,
  'lowest-common-ancestor': lowestCommonAncestor,
  'serialize-deserialize-tree': serializeDeserializeTree,
  'flatten-tree-to-list': flattenTreeToList,
  'right-side-view': rightSideView,
  'word-search-ii': wordSearchII,
}

export function getTraces(problemId: string): AlgoTrace[] | undefined {
  return traces[problemId]
}
