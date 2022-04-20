#include <algorithm>
#include <iterator>

namespace mx {

template <typename FwIter, typename Func>
void adjacent_pair_impl(FwIter first, FwIter last, Func f,
                        std::forward_iterator_tag /*tag*/) {
  if (first != last) {
    auto trailer = first;
    ++first;
    for (; first != last; ++first, ++trailer) {
      f(*trailer, *first);
    }
  }
}

template <typename InIter, typename Func>
void adjacent_pair_impl(InIter first, InIter last, Func f,
                        std::input_iterator_tag /*tag*/) {
  if (first != last) {
    auto trailer = *first;
    ++first;
    for (; first != last; ++first) {
      f(trailer, *first);
      trailer = *first;
    }
  }
}

template <typename It, typename Func>
void adjacent_pair(It first, It last, Func f) {
  return adjacent_pair_impl(
      first, last, f, typename std::iterator_traits<It>::iterator_category{});
}

template <typename FwIter, typename Func>
void for_all_pairs(FwIter first, FwIter last, Func f) {
  if (first != last) {
    auto trailer = first;
    ++first;
    for (; first != last; ++first, ++trailer) {
      for (auto it = first; it != last; ++it) {
        f(*trailer, *it);
      }
    }
  }
}

template <typename InIter, typename OutIter, typename Pred>
auto copy_while(InIter first, InIter last, OutIter result, Pred p)
    -> std::pair<InIter, OutIter> {
  while (first != last && p(*first)) {
    *result++ = *first++;
  }
  return {first, result};
}

template <typename InIter, typename T, typename Func>
void split(InIter first, InIter last, T const &t, Func f) {
  while (true) {
    auto found = std::find(first, last, t);
    f(first, found);
    if (found == last) {
      break;
    }
    first = ++found;
  }
}

template <typename It>
auto cut_paste(It cut_begin, It cut_end, It paste_begin) -> std::pair<It, It> {
  if (paste_begin < cut_begin) {
    return {paste_begin, std::rotate(paste_begin, cut_begin, cut_end)};
  }
  if (cut_end < paste_begin) {
    return {std::rotate(cut_begin, cut_end, paste_begin), paste_begin};
  }
  return {cut_begin, cut_end};
}
} // namespace mx
