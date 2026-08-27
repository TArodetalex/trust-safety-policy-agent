# Marketplace Intellectual Property and Deceptive Commerce Policy

Version: `v1.0.0`

This synthetic policy is created for public demonstration and evaluation. It is
not copied from any private or production policy.

## POL-CF-001 Counterfeit

Reject content that offers, promotes, or requests goods represented as genuine
branded products when the text indicates that they are fake, replica, 1:1,
unauthorized copies, or otherwise not authentic.

Strong signals include:

- `fake`, `counterfeit`, `replica`, `1:1`, `mirror copy`, or `AAA copy`;
- use of a protected brand together with an explicit non-authenticity signal;
- concealment language intended to evade authenticity checks.

Do not infer counterfeit status from low price, country of manufacture, or a
brand mention alone.

## POL-KO-001 Knockoff

Reject content that markets a product as an imitation of a protected design,
trade dress, or branded style without claiming the item is genuine.

Strong signals include:

- `dupe`, `knockoff`, `inspired copy`, or `same design as`;
- intentional imitation of a distinctive design combined with a brand reference;
- promotion of an unlicensed lookalike as a substitute for the branded product.

Generic style terms such as `luxury style` are insufficient without a protected
brand or a clearly distinctive design reference.

## POL-TM-001 Trademark Misuse

Reject commercial content that uses a protected trademark in a misleading,
unauthorized, or promotional way likely to create confusion about source,
sponsorship, or affiliation.

Examples include:

- placing a brand logo on an unrelated generic product;
- using a brand as the primary product identity when the product is not branded;
- claiming authorization, sponsorship, or collaboration without support.

Incidental background exposure and accurate compatibility descriptions are
covered by exemptions below.

## POL-SI-001 Shop Impersonation

Reject shop names, avatars, biographies, or promotional content that falsely
present a seller as an official, flagship, authorized, or verified brand store.

Strong signals include:

- `official store`, `flagship`, `authorized dealer`, or `verified outlet`;
- a brand logo used as the shop's primary identity;
- wording that a reasonable buyer would interpret as brand ownership or approval.

A shop may truthfully describe the products it resells, but it must not imply
that the shop itself is operated or endorsed by the brand.

## POL-RQ-001 Risky Query

Reject search queries or requests that explicitly seek counterfeit goods,
knockoffs, unauthorized branded copies, or methods for evading platform
authenticity controls.

Examples include:

- `where to buy fake [brand]`;
- `best [brand] dupe`;
- `how to pass authenticity check for replica goods`.

Neutral informational queries, authenticity checks, news, repair, or
compatibility research are allowed.

## POL-EX-001 Exemptions

Approve content when one of these exemptions applies and no separate violation
signal is present:

### Compatibility

Accurate phrases such as `compatible with`, `fits`, or `for use with` may name a
brand only to explain interoperability. The wording must not imply manufacture,
sponsorship, or authorization by that brand.

### Second-hand

Truthful resale of an authentic used product is allowed when the listing clearly
states its used or pre-owned condition and does not contain counterfeit signals.

### Meaningful word

A trademark that is also an ordinary dictionary word may be used in its normal
non-trademark meaning, such as `apple juice`, `gap year`, or `shell necklace`.

### Incidental exposure

A logo or branded product appearing incidentally in a person, room, street, or
background image is allowed when it is not the promoted subject.

### Co-brand

A documented collaboration or licensed co-brand may use both brands. Unsupported
claims such as `official collaboration` remain violations.

## Evidence and Escalation

A rejection must cite a directly relevant policy passage. If retrieved evidence
is weak, conflicting, or insufficient, the automated system must return
`need_review` rather than inventing a rule or forcing a rejection.
