import type { Problem } from '../types'

import binaryTreeMaxPathSum from './binary-tree-max-path-sum'
import climbingStairs from './climbing-stairs'
import combinationSum from './combination-sum'
import containerWithMostWater from './container-with-most-water'
import courseSchedule from './course-schedule'
import dailyTemperatures from './daily-temperatures'
import editDistance from './edit-distance'
import findAnagrams from './find-anagrams'
import findMedianDataStream from './find-median-data-stream'
import findPeakElement from './find-peak-element'
import flattenTreeToList from './flatten-tree-to-list'
import houseRobber from './house-robber'
import implementTrie from './implement-trie'
import inorderTraversal from './inorder-traversal'
import kthLargestElement from './kth-largest-element'
import largestRectangleHistogram from './largest-rectangle-histogram'
import levelOrderTraversal from './level-order-traversal'
import linkedListCycleIi from './linked-list-cycle-ii'
import longestCommonSubsequence from './longest-common-subsequence'
import longestIncreasingSubsequence from './longest-increasing-subsequence'
import longestPalindromicSubstring from './longest-palindromic-substring'
import longestSubstringWithoutRepeating from './longest-substring-without-repeating'
import lowestCommonAncestor from './lowest-common-ancestor'
import lruCache from './lru-cache'
import medianTwoSortedArrays from './median-two-sorted-arrays'
import mergeKSortedLists from './merge-k-sorted-lists'
import minimumWindowSubstring from './minimum-window-substring'
import nQueens from './n-queens'
import numberOfIslands from './number-of-islands'
import permutations from './permutations'
import productExceptSelf from './product-except-self'
import reverseLinkedList from './reverse-linked-list'
import rightSideView from './right-side-view'
import rottingOranges from './rotting-oranges'
import searchRotatedArray from './search-rotated-array'
import serializeDeserializeTree from './serialize-deserialize-tree'
import singleNumber from './single-number'
import subarraySumEqualsK from './subarray-sum-equals-k'
import threeSum from './three-sum'
import topKFrequent from './top-k-frequent'
import trappingRainWater from './trapping-rain-water'
import twoSum from './two-sum'
import uniquePaths from './unique-paths'
import wordBreak from './word-break'
import wordSearch from './word-search'
import wordSearchIi from './word-search-ii'

export const problems: Record<string, Problem> = {
  'binary-tree-max-path-sum': binaryTreeMaxPathSum,
  'climbing-stairs': climbingStairs,
  'combination-sum': combinationSum,
  'container-with-most-water': containerWithMostWater,
  'course-schedule': courseSchedule,
  'daily-temperatures': dailyTemperatures,
  'edit-distance': editDistance,
  'find-anagrams': findAnagrams,
  'find-median-data-stream': findMedianDataStream,
  'find-peak-element': findPeakElement,
  'flatten-tree-to-list': flattenTreeToList,
  'house-robber': houseRobber,
  'implement-trie': implementTrie,
  'inorder-traversal': inorderTraversal,
  'kth-largest-element': kthLargestElement,
  'largest-rectangle-histogram': largestRectangleHistogram,
  'level-order-traversal': levelOrderTraversal,
  'linked-list-cycle-ii': linkedListCycleIi,
  'longest-common-subsequence': longestCommonSubsequence,
  'longest-increasing-subsequence': longestIncreasingSubsequence,
  'longest-palindromic-substring': longestPalindromicSubstring,
  'longest-substring-without-repeating': longestSubstringWithoutRepeating,
  'lowest-common-ancestor': lowestCommonAncestor,
  'lru-cache': lruCache,
  'median-two-sorted-arrays': medianTwoSortedArrays,
  'merge-k-sorted-lists': mergeKSortedLists,
  'minimum-window-substring': minimumWindowSubstring,
  'n-queens': nQueens,
  'number-of-islands': numberOfIslands,
  'permutations': permutations,
  'product-except-self': productExceptSelf,
  'reverse-linked-list': reverseLinkedList,
  'right-side-view': rightSideView,
  'rotting-oranges': rottingOranges,
  'search-rotated-array': searchRotatedArray,
  'serialize-deserialize-tree': serializeDeserializeTree,
  'single-number': singleNumber,
  'subarray-sum-equals-k': subarraySumEqualsK,
  'three-sum': threeSum,
  'top-k-frequent': topKFrequent,
  'trapping-rain-water': trappingRainWater,
  'two-sum': twoSum,
  'unique-paths': uniquePaths,
  'word-break': wordBreak,
  'word-search': wordSearch,
  'word-search-ii': wordSearchIi,
}

export function getProblem(id: string): Problem | undefined {
  return problems[id]
}
