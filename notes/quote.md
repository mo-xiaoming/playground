Because you’re a C++ programmer, there’s an above-average chance you’re a performance freak. If you’re not, you’re still probably sympathetic to their point of view. (If you’re not at all interested in performance, shouldn’t you be in the Python room down the hall?) -- Scott Meyers, "Effective Modern C++"

Because, as the saying goes, in theory, there’s no difference between theory and practice, but in practice, there is. -- Scott Meyers, "Effective Modern C++"?

Quote from Jon Bentley, "Programming Pearls", page 35, 36 (slightly re-worded to use C syntax):

I've assigned this problem [binary search] in courses at Bell Labs and
IBM.  Professional programmers had a couple of hours to convert the
above description [of binary search] into a program in the language of
their choice.....at the end of the period, most programmers reported
that they had written correct code for the task.  We would then take
30 minutes to examine their code....In several cases, and with over
100 programmers, the results varied little.  90 % of the programmers
found bugs in their programs.


“A really bad idea, embraced by millions of people, is still a really bad idea.” ~ Tony Blauer


“Death is like being stupid. It’s only painful for other people” - Ricky Gervais 2018


Aldous Huxley - Fifth Philosopher's Song

A million million spermatozoa
All of them alive;
Out of their cataclysm but one poor Noah
Dare hope to survive.

And among that billion minus one
Might have chanced to be
Shakespeare, another Newton, a new Donne—
But the One was Me.

Shame to have ousted your betters thus,
Taking ark while the others remained outside!
Better for all of us, froward Homunculus,
If you’d quietly died!


"What's it like after you die?"
"What's it like 13 and half billions year before you were born, it's probably like that" - Ricky Gervais 2018

"Why don't you pray just incase there is a God"
"Why don't you hang garlic over your door just incase there is a Dracula" - Ricky Gervais 2018
"It has nothing to do with the credibility of the truth, as to do with the popularity of the idea"

A committee is a group that keeps minutes and loses hours - Milton Berle
If you want to destroy any idea in the world, get a committee working on it -- Kettering
It's hard to design product by focus groups -- Steven Jobs

"This was a couple of software engineers who put this in for whatever reason," Michael Horn, VW's U.S. head, said about the software code designed to cheat on emissions tests, which the company put in diesel cars since 2009.

One programmer's "clean code" is another programmer's "over-engineering".

Architecture is the stuff you can't Google -- @markrichardssa

The only concrete process in the Agile Manifesto is the retrospective. The only essential characteristic of an Agile org is continuous improvement and learning.


Hardware engineer: If it ain't broke, don't fix it.
Software engineer: If it ain't broke, it doesn't have enough features yet. -- Jon Kalb

“A complex system that works is invariably found to have evolved from a simple system that worked. ... A complex system designed from scratch never works & cannot be made to work. You have to start over, beginning with a working simple system.”  -- Gall's Law

## Allen Holub

Don't ask "what do you need", Instead, ask "what are the most significant problems you're facing"

We are not building houses. I don't know how many times I've heard "you can't build a house like that" used as a way to blow off incremental development, no estimates, etc.. We are not building houses.


Don't get caught in the flexibility trap. Flexibility usually comes with added complexity that that handles eventualities that never happen. Instead of "flexible", I'd go with "simple". If it doesn't work out, just replace it.


Guessing about the way a 'general' API looks is great way to produce over-complex junk. Talk to customers & build exactly what they need, expanding only as necessary as they need, expanding only as necessary as you pull more customers in. You'll get a *much* simpler/better API than if you try to guess what they need

Incremental dev is a risk-reduction strategy. Build small, release, improve based on the feedback. Estimates, milestones, up-front design is another risk-reduction strategy. Build big. Release big. Incremental dev yields something salable very quickly, so ROI is quicker & risk lower

"T-shaped skills" (deep knowledge of a few things, broad knowledge of others) are an essential characteristic for a team member. The goal is, if somebody's out for some reason, work can continue. That does not mean that everybody knows everything, but there has to be enough overlap that work never comes to a standstill. T-shaped skills also make it a lot easier to "swarm"—to work together effectively on a task where people are stuck.


Pretty much all you need for agility is common sense. It’s interesting how the things we learn about how stuff is _supposed_ to work gets in the way of that :-).
@johncutlefish
“That just sounds like common sense...what exactly is the way?”

“[common sense]”

“No no... the actual way. It can’t just be that?”

“OK... here’s another approach. Create an environment where common sense can thrive.”


You can’t waste the time of those highly-paid developers doing something as mundane as testing, so let’s throw the tests to low-paid “testers.” That naturally leads to thinking of “testing” as a separate skill from development rather than being an integral part of it, and that leads to thinking that “testers” have some magical ability to see things and think of issues that the developers can’t conceive of. I see that more as an attempt at job security than any sort of reality. Good devs can see the issues just fine, and if the dev’s aren’t good, you can train them. That testers-as-second-class-citizens thing also leads to the mistaken assumption that testing isn’t nearly as important as slinging code. In fact, if you don’t test and code at the same time, the coding goes much more slowly. Testing and coding are inseparable. You can’t do one without the other, and both must be done concurrently. The next problem is thinking that software dev is done on an assembly line, where you can’t test most things until the product is “finished,” so there’s s QA step at the end of the line. That led to a throw-it-over-the-wall approach to testing, which never worked. The idea of testing being separate from dev comes, I think, from a few false assumptions (and is tightly coupled with waterfall thinking). The first assumption is that testing is a lesser art than programming. A separate QA dept. was always a bottleneck that slowed down everything. Nonetheless, people who should know better, still hold on to that concept, saying that some things (e.g. performance or integration) are better handled by throwing the code over the wall. I just don’t see it.  Bottlenecks are bottlenecks. Problems discovered later rather than earlier are much more expensive to fix. IMO, testing and dev must go on simultaneously, and be done by the same people. You don’t have time for silos. QA is a skill, not a job title.

The best way to build trust is to release often. This is true both with external customers/clients and with people inside the organization, up or down the hierarchy.

The thing you release today my render all or part of your backlog meaningless. The "domain" encompasses the way that people work. Stories, in fact, describe people working in the domain. When you release, you change the way people work. Consequently, understanding the domain is a continuous loop. Every time you release something valuable into the domain, that release changes the way people work. Stories on the backlog that assume that people work in the old way will be useless. The best solution is to keep the backlog as short as possible (or eliminate it altogether).

A 12-month road map is like a map through 1000 miles of sand dunes. Good luck with that.

The only acceptable known bug count at release is zero. If that's not your policy now, institute it immediately. You'll never get a handle on "tech debt" if you keep adding to it.

Someone who's arguing "we don't have the money or people to replace this mess" is actually arguing "we don't have the money or people to stay in business."

"Bugs are code that doesn’t behave the way you expect, as compared to tech debt, where the code behaves exactly as you expect, but your expectations are incorrect." -- Allen Holub

A jaded coach used to say, “the *correct* estimate is the highest number the sponsor won’t refuse.”
